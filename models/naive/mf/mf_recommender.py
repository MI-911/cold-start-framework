from typing import Dict, List

from models.shared.base_recommender import RecommenderBase
from models.naive.mf.mf import MatrixFactorisation
import numpy as np
import random
from loguru import logger

from models.shared.meta import Meta
from models.shared.user import WarmStartUser
from shared.utility import get_combinations


def flatten_dataset(training: Dict[int, WarmStartUser]):
    t, v = [], []
    for u, data in training.items():
        for e, r in data.training.items():
            t.append((u, e, r))

        v.append((u, (data.validation['positive'], data.validation['negative'])))

    return t, v


class MatrixFactorisationRecommender(RecommenderBase):

    # MF doesn't conduct an interview
    def interview(self, answers: Dict, max_n_questions=5) -> List[int]:
        return [0]

    def get_parameters(self):
        return self.optimal_params

    def load_parameters(self, params):
        self.optimal_params = params

    def __init__(self, meta: Meta, use_cuda=False):
        super(MatrixFactorisationRecommender, self).__init__(meta, use_cuda)
        self.meta = meta
        self.optimal_params = None

    def convert_rating(self, rating):
        if rating == 1:
            return 1
        elif rating == -1:
            return 0
        elif rating == 0:
            # We can make a choice here - either return 0 or 0.5
            return 0.5

    def batches(self, triples, n=64):
        for i in range(0, len(triples), n):
            yield triples[i:i + n]

    def warmup(self, training: Dict[int, WarmStartUser], interview_length=5) -> None:
        hit_rates = []
        training, validation = flatten_dataset(training)

        if self.optimal_params is None:
            parameters = {
                'k': [1, 2, 5, 10, 15, 25, 50]
            }
            for params in get_combinations(parameters):
                logger.debug(f'Fitting MF with params: {params}')
                self.model = MatrixFactorisation(len(self.meta.users),
                                                 len(self.meta.entities),
                                                 params['k'])

                hit_rates.append((self._fit(training, validation, max_iterations=100), params))

            hit_rates = sorted(hit_rates, key=lambda x: x[0], reverse=True)
            _, best_params = hit_rates[0]

            self.optimal_params = best_params

            self.model = MatrixFactorisation(len(self.meta.users),
                                             len(self.meta.entities),
                                             self.optimal_params['k'])
            logger.info(f'Found best parameters for MF: {self.optimal_params}')
            self._fit(training, validation)
        else:
            self.model = MatrixFactorisation(len(self.meta.users),
                                             len(self.meta.entities),
                                             self.optimal_params['k'])
            self._fit(training, validation)

    def _fit(self, training, validation, max_iterations=100, verbose=True, save_to='./'):
        validation_history = []

        for epoch in range(max_iterations):
            random.shuffle(training)
            self.model.train_als(training)

            if epoch % 10 == 0:
                ranks = []
                for user, (pos, negs) in validation:
                    predictions = self.model.predict(user, negs + [pos])
                    predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
                    predictions = {i: rank for rank, (i, s) in enumerate(predictions)}
                    ranks.append(predictions[pos])

                _hit = np.mean([1 if r < 10 else 0 for r in ranks])
                validation_history.append(_hit)

                if verbose:
                    logger.info(f'Hit@10 at epoch {epoch}: {np.mean([1 if r < 10 else 0 for r in ranks])}')

        return np.mean(validation_history[-10:])

    def predict(self, items, answers):
        # User is the average user embedding
        return self.model.predict_avg(items)
