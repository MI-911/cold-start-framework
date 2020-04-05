from concurrent.futures import wait
from concurrent.futures.process import ProcessPoolExecutor
from concurrent.futures.thread import ThreadPoolExecutor
from copy import deepcopy
from typing import List, Dict

from loguru import logger
from scipy.sparse import csr_matrix
from sklearn.neighbors._dist_metrics import DistanceMetric

from recommenders.base_recommender import RecommenderBase
from shared.user import WarmStartUser
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise

class KNNRecommender(RecommenderBase):
    def __init__(self, meta):
        RecommenderBase.__init__(self, meta=meta)
        self.data = None
        self.model = NearestNeighbors
        self.optimal_params = None

        # Utility information
        self.user_mean = None
        self.entity_mean = None

        # Model parameters
        self.k = 5
        self.metric = None
        self.user_bias = False
        self.entity_bias = False

    def _get_param_combinations(self):
        params = []
        for k in [5, 10, 15, 20]:
            for m in ['cosine']:
                for ub in [True, False]:
                    for eb in [False]:
                        params.append({'k': k, 'metric': m, 'user_bias': ub, 'entity_bias': eb})
        return params

    def _set_params(self, params: Dict):
        self.k = params['k']
        self.metric = params['metric']
        self.user_bias = params['user_bias']
        self.entity_bias = params['entity_bias']

    def fit(self, training: Dict[int, WarmStartUser]):
        logger.debug('Building data')
        ratings_matrix = np.zeros((len(self.meta.users), len(self.meta.entities)))
        for user, data in training.items():
            for entity, rating in data.training.items():
                ratings_matrix[user, entity] = rating

        self.user_mean = np.mean(ratings_matrix, axis=0)
        self.movie_mean = np.mean(ratings_matrix, axis=1)

        mat_movie_features = csr_matrix(ratings_matrix)
        self.data = mat_movie_features

        logger.debug('Fitting')
        best_params = None
        best_hr = -1
        for params in self._get_param_combinations():
            logger.debug(f'Trying with parameters: {params}')
            self._set_params(params)
            self.model = NearestNeighbors(n_neighbors=self.k, metric=self.metric)
            self.model.fit(self.data)
            hr = self._multi_fit(training)
            if hr > best_hr:
                logger.debug(f'Found better parameters: {params}, HR@10: {hr}')
                best_hr = hr
                best_params = deepcopy(params)

        self.optimal_params = best_params
        self._set_params(best_params)
        self.model = NearestNeighbors(n_neighbors=self.k, metric=self.metric)
        self.model.fit(self.data)

    def _multi_fit(self, training: Dict[int, WarmStartUser], workers=8):
        lst = list(training.items())
        chunks = [lst[i::workers] for i in range(workers)]

        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = []
            for chunk in chunks:
                futures.append(executor.submit(self._fit, dict(chunk)))

            wait(futures)

        return sum([f.result() for f in futures]) / float(len(training))

    def _fit(self, training: Dict[int, WarmStartUser]):
        hits = 0.
        for user, warm in training.items():
            pos, neg = warm.validation.values()
            items = neg + [pos]
            item_scores = []
            for item in items:
                item_scores.append((item, self._predict(self.data[user], item)))

            item_scores = sorted(item_scores, key=lambda s: s[1], reverse=True)[:10]
            if pos in [i for i, _ in item_scores]:
                hits += 1.

        return hits

    def _score_function(self, users, similarities, ratings):
        if self.user_bias:
            ratings = ratings - self.user_mean[users]

        score = np.sum(ratings * similarities) / np.sum(similarities)

        return score

    def _predict(self, user_ratings: csr_matrix, item: int) -> int:
        similarities, users = self.model.kneighbors(user_ratings, n_neighbors=len(self.meta.users))
        user_sim = list(zip(users[0], similarities[0]))

        local_users = []
        local_similarities = []
        local_ratings = []

        users_rated_item = list(self.data[:, item].nonzero()[0])

        for user, similarity in user_sim:
            if user in users_rated_item:
                local_users.append(user)
                local_similarities.append(similarity)
                local_ratings.append(self.data[user, item])

                if len(local_users) >= self.k:
                    break

        local_users = np.array(local_users)
        local_similarities = np.array(local_similarities)
        local_ratings = np.array(local_ratings)

        # In case of unrated item
        if len(local_users) > 0:
            score = self._score_function(local_users, local_similarities, local_ratings)
        else:
            score = -1

        return score

    def predict(self, items: List[int], answers: Dict[int, int]) -> Dict[int, float]:
        user_ratings = np.zeros((len(self.meta.entities),))
        for itemID, rating in answers.items():
            user_ratings[itemID] = rating
        user_ratings = csr_matrix(user_ratings)

        item_scores = {}
        for item in items:
            item_scores[item] = self._predict(user_ratings, item)

        return item_scores
