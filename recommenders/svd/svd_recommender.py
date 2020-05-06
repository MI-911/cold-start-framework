import operator
from typing import List, Dict

from loguru import logger
from sklearn.utils.extmath import randomized_svd
from tqdm import tqdm

from recommenders.base_recommender import RecommenderBase
from shared.meta import Meta
from shared.user import WarmStartUser
from shared.utility import csr, get_combinations

import scipy
import numpy as np


class SVDRecommender(RecommenderBase):
    def __init__(self, meta: Meta):
        super().__init__(meta)
        self.ratings = None
        self.U = None
        self.V = None
        self.optimal_params = None

    def _recommend(self, users):
        return np.dot(self.U[users, :], self.V.T)

    def _fit(self, training, factors):
        self.ratings = csr(training, {-1: 1, 0: 3, 1: 5})

        U, sigma, VT = randomized_svd(self.ratings, factors)
        sigma = scipy.sparse.diags(sigma, 0)
        self.U = U * sigma
        self.V = VT.T

    def fit(self, training: Dict[int, WarmStartUser]):
        if not self.optimal_params:
            parameters = {
                'factors': [20, 40, 60, 80]
            }

            combinations = get_combinations(parameters)
            logger.debug(f'{len(combinations)} hyperparameter combinations')

            results = list()
            for combination in combinations:
                logger.debug(f'Trying {combination}')

                self._fit(training, **combination)

                predictions = list()
                for u_idx, (_, user) in tqdm(list(enumerate(training.items()))):
                    prediction = self._predict(u_idx, user.validation.to_list())

                    predictions.append((user.validation, prediction))

                score = self.meta.validator.score(predictions, self.meta)
                results.append((combination, score))

                logger.info(f'Score: {score}')

            best = sorted(results, key=operator.itemgetter(1), reverse=True)[0][0]
            logger.info(f'Found best: {best}')

            self.optimal_params = best
        else:
            logger.debug(f'Using stored hyperparameters {self.optimal_params}')

        self._fit(training, **self.optimal_params)

    def _predict(self, user, items):
        scores = self._recommend(user)
        item_scores = sorted(list(enumerate(scores)), key=operator.itemgetter(1), reverse=True)

        return {index: score for index, score in item_scores if index in items}

    def predict(self, items: List[int], answers: Dict[int, int]) -> Dict[int, float]:
        # magic
        pass
