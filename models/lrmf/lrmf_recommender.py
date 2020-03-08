import pickle
from abc import ABC
from typing import Dict, List
from models.lrmf.lrmf import LRMF
from models.lrmf.decision_tree import LIKE, DISLIKE
from models.shared.base_recommender import RecommenderBase
import numpy as np
import pickle
from loguru import logger

from shared.utility import get_combinations


def get_rating_matrix(training, n_users, n_entities, rating_map=None):
    if not rating_map:
        rating_map = {
            1: LIKE,
            0: DISLIKE,
            -1: DISLIKE
        }

    R = np.zeros((n_users, n_entities))
    for user, data in training.items():
        for entity, rating in data.training.items():
            R[user, entity] = rating_map[rating]

    return R


def choose_candidates(rating_matrix, n=100):
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

        scores = model.validate(user, to_val)
        sorted_scores = sorted([(item, score) for item, score in scores.items()], key=lambda x: x[1], reverse=True)
        top_items = [item for item, score in sorted_scores][:10]
        hits.append(1 if pos in top_items else 0)

    return np.mean(hits)


class LRMFRecommender(RecommenderBase):
    def __init__(self, meta):
        super(LRMFRecommender, self).__init__(meta)
        self.meta = meta
        self.model = None

        self.n_candidates = 100

        self.n_users = len(meta.users)
        self.n_entities = len(meta.entities)

        self.params = None
        self.best_model = None
        self.best_hit = 0

    def warmup(self, training: Dict) -> None:
        self.best_model = None
        self.best_hit = 0

        if not self.params:
            for params in get_combinations({
                'kk': [1, 2, 5, 10, 20],
                'reg': [0.001, 0.0001, 0.00001]
            }):
                logger.info(f'Fitting LRMF with params {params}')

                self.model = LRMF(
                    n_users=self.n_users,
                    n_entities=self.n_entities,
                    l1=3, l2=3,
                    kk=params['kk'],
                    alpha=params['reg'],
                    beta=params['reg'])

                self._fit(training)

            self.model = self.best_model
            self.params = {'kk': self.model.kk}

        else:
            self.model = LRMF(
                n_users=self.n_users,
                n_entities=self.n_entities,
                l1=3, l2=3,
                kk=self.params['kk'],
                alpha=self.params['reg'],
                beta=self.params['reg'])

            self._fit(training)

    def _fit(self, training):
        R = get_rating_matrix(training, self.n_users, self.n_entities, {
            1: 1,
            0: 0,
            -1: 0
        })

        candidates = choose_candidates(R, n=100)

        for i in range(10):
            logger.debug(f'LRMF starting iteration {i}')
            self.model.fit(R, candidates)
            hit = validate_hit(self.model, training)
            logger.debug(f'Training: {hit} Hit@10')
            if hit > self.best_hit:
                logger.debug(f'LRMF found new best model at {hit} Hit@10')
                self.best_hit = hit
                self.best_model = pickle.loads(pickle.dumps(self.model))  # Save the model

        self.model = self.best_model

    def interview(self, answers: Dict[int, int], max_n_questions=5) -> List[int]:
        return self.model.get_next_question(answers)

    def predict(self, items: List[int], answers: Dict[int, int]) -> Dict[int, float]:
        return self.model.rank(items, answers)

    def get_parameters(self):
        return self.params

    def load_parameters(self, params):
        self.params = params


if __name__ == '__main__':
    training = pickle.load(open('../../partitioners/data/training.pkl', 'rb'))
    testing = pickle.load(open('../../partitioners/data/testing.pkl', 'rb'))
    meta = pickle.load(open('../../partitioners/data/meta.pkl', 'rb'))

    recommender = LRMFRecommender(meta)
    recommender.warmup(training)
