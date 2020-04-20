import multiprocessing
from concurrent.futures import wait
from concurrent.futures.process import ProcessPoolExecutor
from concurrent.futures.thread import ThreadPoolExecutor
from copy import deepcopy
from typing import List, Dict

from loguru import logger
from scipy.sparse import csr_matrix

from recommenders.base_recommender import RecommenderBase
from shared.user import WarmStartUser
import numpy as np
from sklearn.neighbors import NearestNeighbors


class KNNRecommender(RecommenderBase):
    def __init__(self, meta):
        RecommenderBase.__init__(self, meta)
        self.optimal_params = None

        # Matrix dimensions
        self.n_xs = len(meta.users)
        self.n_ys = len(meta.entities)

        # Numpy arrays
        self.mean_centered_ratings = np.zeros((len(meta.users),))
        self.entity_vectors = np.zeros((self.n_xs, self.n_ys))
        self.plain_entity_vectors = np.zeros((self.n_xs, self.n_ys))
        self.pearson_entity_vectors = np.zeros((self.n_xs, self.n_ys))

        # Model parameters
        self.k = 5
        self.metric = None

    def _cosine_similarity(self, user_vectors, user_k, eps=1e-8):
        user_sim_vec = self.entity_vectors[user_k]
        top = np.einsum('i,ji->j', user_vectors, user_sim_vec)
        samples_norm = np.sqrt(np.sum(user_vectors ** 2, axis=0))
        entity_norm = np.sqrt(np.sum(user_sim_vec ** 2, axis=1))
        bottom = np.maximum(samples_norm * entity_norm, eps)

        return top / bottom

    @staticmethod
    def chunkify(lst, chunks):
        return [lst[i::chunks] for i in range(chunks)]

    @staticmethod
    def _get_param_combinations():
        params = []
        for k in [5, 10, 15, 20]:
            for m in ['cosine', 'pearson']:
                    params.append({'k': k, 'metric': m})
        return params

    def _set_params(self, params: Dict):
        self.k = params['k']
        self.metric = params['metric']
        if params['metric'] == 'cosine':
            self.entity_vectors = self.plain_entity_vectors.copy()
        elif params['metric'] == 'pearson':
            self.entity_vectors = self.pearson_entity_vectors.copy()

    def fit(self, training: Dict[int, WarmStartUser]):
        logger.debug('Building data')
        for user, warm in training.items():
            for entity_id, rating in warm.training.items():
                self.plain_entity_vectors[user][entity_id] = rating

        # Calculate user mean.
        for user, _ in training.items():
            self.mean_centered_ratings[user] = np.mean(self.plain_entity_vectors[user])

        # Set adjusted vectors
        for entity in range(self.n_xs):
            indices = np.where(self.plain_entity_vectors[:, entity] != 0)[0]
            for user in indices:
                self.pearson_entity_vectors[user][entity] = self.plain_entity_vectors[user][entity] - \
                                                            self.mean_centered_ratings[user]

        logger.debug('Fitting')
        if self.optimal_params is None:
            best_params = None
            best_hr = -1
            for params in self._get_param_combinations():
                logger.debug(f'Trying with parameters: {params}')
                self._set_params(params)
                hr = self._multi_fit(training, workers=multiprocessing.cpu_count())
                if hr > best_hr:
                    logger.debug(f'Found better parameters: {params}, score: {hr}')
                    best_hr = hr
                    best_params = deepcopy(params)

            self.optimal_params = best_params

        self._set_params(self.optimal_params)

    def _multi_fit(self, training: Dict[int, WarmStartUser], workers=8):
        lst = list(training.items())
        chunks = [lst[i::workers] for i in range(workers)]

        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = []
            for chunk in chunks:
                futures.append(executor.submit(self._fit, dict(chunk)))

            wait(futures)

        predictions = [t for future in futures for t in future.result()]
        score = self.meta.validator.score(predictions, self.meta)

        return score

    def _fit(self, training: Dict[int, WarmStartUser]):
        predictions = []
        for user, warm in training.items():
            preds = {}
            for item in warm.validation.to_list():
                preds[item] = self._predict(self.entity_vectors[user], item)

            predictions.append((warm.validation, preds))
        return predictions

    def _score_function(self, users, similarities, ratings):
        if self.user_bias:
            ratings = ratings - self.user_mean[users]

        score = np.sum(ratings * similarities) / np.sum(similarities)

        return score

    def _predict(self, user_vector, item: int):
        related = np.where(self.entity_vectors[:, item] != 0)[0]

        if related.size == 0:
            return 0

        cs = self._cosine_similarity(user_vector, related)

        top_k = sorted([(r, s) for r, s in zip(related, cs)], key=lambda x: x[1], reverse=True)[:self.k]
        ratings = [(self.entity_vectors[i][item], sim) for i, sim in top_k]
        return np.einsum('i,i->', *zip(*ratings))

    def predict(self, items: List[int], answers: Dict[int, int]) -> Dict[int, float]:
        user_ratings = np.zeros((len(self.meta.entities),))
        for itemID, rating in answers.items():
            user_ratings[itemID] = rating

        score = {}
        for item in items:
            score[item] = self._predict(user_ratings, item)

        # A high score means item knn is sure in a positive prediction.
        return score
