from typing import List, Dict

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
from surprise import NormalPredictor
from surprise import Dataset
from surprise import Reader
from surprise import KNNBasic

class KNNRecommender(RecommenderBase):
    def __init__(self, meta):
        RecommenderBase.__init__(self, meta=meta)
        self.k = 5
        self.metric = None
        self.data = None
        self.model = NearestNeighbors
        self.vector_length = 0
        self.optimal_params = None

    def fit(self, training: Dict[int, WarmStartUser]):

        ratings_matrix = np.zeros((len(self.meta.users), len(self.meta.entities)))
        for user, data in training.items():
            for entity, rating in data.training.items():
                ratings_matrix[user, entity] = rating

        mat_movie_features = csr_matrix(ratings_matrix)
        self.data = mat_movie_features
        self.model = NearestNeighbors()
        self.model.fit(self.data)

    def predict(self, items: List[int], answers: Dict[int, int]) -> Dict[int, float]:
        sparse = np.zeros((len(self.meta.entities),))
        for itemID, rating in answers.items():
            sparse[itemID] = rating
        sparse = csr_matrix(sparse)
        similarities, users = self.model.kneighbors(sparse, n_neighbors=len(self.meta.users))
        user_sim = list(zip(users[0], similarities[0]))

        s_coo = self.data.tocoo()
        nonzero = set(zip(s_coo.row, s_coo.col))

        item_scores = {}
        for item in items:
            local_users = []
            local_similarities = []
            local_ratings = []

            for user, similarity in user_sim:
                if (user, item) in nonzero:
                    local_users.append(user)
                    local_similarities.append(similarity)
                    local_ratings.append(self.data[user, item])

                    if len(local_users) > self.k:
                        break

            local_users = np.array(local_users)
            local_similarities = np.array(local_similarities)
            local_ratings = np.array(local_ratings)

            # In case of unrated item
            if len(local_users) > 0:
                score = np.sum(local_ratings * local_similarities) / np.sum(local_similarities)
            else:
                score = -1

            item_scores[item] = score

        return item_scores

    def temp(self):
        # train_df = pd.DataFrame(ratings_dict)
        # reader = Reader(rating_scale=(-1, 1))
        #
        # train_data = Dataset.load_from_df(train_df[['userID', 'itemID', 'rating']], reader=reader)
        #
        # train_data = train_data.build_full_trainset()
        #
        # self.knn = KNNBasic()
        # self.knn.fit(train_data)
        # hits = 0
        # for user, data in training.items():
        #     pos, neg = data.validation.values()
        #     scores = []
        #     for item in [pos] + neg:
        #         pred = self.knn.predict(user, item)
        #         scores.append((item, pred.est))
        #
        #     top = sorted(scores, key=lambda x: x[1], reverse=True)[:10]
        #     hits += 1 if pos in [item for item, _ in top] else 0
        #
        # res = hits / len(training) if hits else 0.0
        # print('test')
        pass