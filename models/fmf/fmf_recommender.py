import pickle
from typing import Dict, List, Union

import numpy as np
from loguru import logger

from models.base_recommender import RecommenderBase
from models.fmf.fmf import LIKE, DISLIKE, UNKNOWN, FMF
from shared.utility import get_combinations


def get_rating_matrix(training, n_users, n_entities, rating_map=None):
    """
    Returns an [n_users x n_entities] ratings matrix.
    """
    if rating_map is None:
        rating_map = {
            1: LIKE,
            0: UNKNOWN,
            -1: DISLIKE
        }

    R = np.ones((n_users, n_entities)) * rating_map[0]
    for user, data in training.items():
        for entity, rating in data.training.items():
            R[user, entity] = rating_map[rating]

    return R


def choose_candidates(rating_matrix, n=100):
    """
    Selects n candidates that can be asked towards in an interview.
    """
    # TODO: Choose candidate items with a mix between popularity and diversity
    n_ratings = rating_matrix.sum(axis=0)
    n_ratings = sorted([(entity, rs) for entity, rs in enumerate(n_ratings)], key=lambda x: x[1], reverse=True)
    return [entity for entity, rs in n_ratings][:n]


def validate_hit(model, training):
    hits = []
    for user, data in training.items():
        pos = data.validation['positive']
        neg = data.validation['negative']
        to_val = neg + [pos]
        np.random.shuffle(to_val)

        scores = model.validate(user, to_val)
        sorted_scores = sorted([(item, score) for item, score in scores.items()], key=lambda x: x[1], reverse=True)
        top_items = [item for item, score in sorted_scores][:10]
        hits.append(1 if pos in top_items else 0)

    return np.mean(hits)


class FMFRecommender(RecommenderBase):
    def __init__(self, meta, use_cuda=False):
        super(FMFRecommender, self).__init__(meta, use_cuda)
        self.meta = meta
        self.model: FMF = Union[FMF, None]

        self.n_candidates = 100

        self.n_users = len(meta.users)
        self.n_entities = len(meta.entities)

        self.params = None
        self.best_model = None
        self.best_hit = 0

    def warmup(self, training: Dict, interview_length=5) -> None:
        self.best_model = None
        self.best_hit = 0

        if not self.params:
            for params in get_combinations({
                'k': [1, 2, 5, 10, 20],
                'reg': [0.01, 0.001, 0.0001]
            }):
                logger.info(f'Fitting FMF with params {params}')

                self.model = FMF(
                    n_users=self.n_users,
                    n_entities=self.n_entities,
                    max_depth=interview_length,
                    n_latent_factors=params['k'],
                    regularization=params['reg']
                )

                self._fit(training)

            self.model = self.best_model
            self.params = {'kk': self.model.kk}

        else:
            self.model = FMF(
                n_users=self.n_users,
                n_entities=self.n_entities,
                max_depth=interview_length,
                n_latent_factors=self.params['k'],
                regularization=self.params['reg']
            )

            self._fit(training)

    def _fit(self, training):
        R = get_rating_matrix(training, self.n_users, self.n_entities)
        candidates = choose_candidates(R, n=100)

        n_iterations = 10
        for i in range(n_iterations):
            self.model.fit(R, candidates)
            hit = validate_hit(self.model, training)

            logger.debug(f'Training iteration {i}: {hit} Hit@10')

            if hit > self.best_hit:
                logger.debug(f'FMF found new best model at {hit} Hit@10')
                self.best_hit = hit
                self.best_model = pickle.loads(pickle.dumps(self.model))  # Save the model

        self.model = self.best_model

    def interview(self, answers: Dict[int, int], max_n_questions=5) -> List[int]:
        return [self.model.interview(answers)]

    def predict(self, items: List[int], answers: Dict[int, int]) -> Dict[int, float]:
        return self.model.rank(items, answers)

    def get_parameters(self):
        return self.params

    def load_parameters(self, params):
        self.params = params
