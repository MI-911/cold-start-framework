import pickle
from typing import Dict, List
from models.lrmf.lrmf import LRMF
from models.lrmf.decision_tree import LIKE, DISLIKE
from models.shared.base_recommender import RecommenderBase
import numpy as np


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
    n_ratings = rating_matrix.sum(axis=0)
    n_ratings = sorted([(entity, rs) for entity, rs in enumerate(n_ratings)], key=lambda x: x[1])
    return [entity for entity, rs in n_ratings][:n]


class LRMFRecommender(RecommenderBase):
    def __init__(self, meta):
        super(LRMFRecommender, self).__init__()
        self.meta = meta
        self.model = None

        self.n_candidates = 100

        self.n_users = len(meta.users)
        self.n_entities = len(meta.entities)

    def warmup(self, training: Dict) -> None:
        R = get_rating_matrix(training, self.n_users, self.n_entities, {
            1: 1,
            0: 0,
            -1: 0
        })

        candidates = choose_candidates(R, n=100)

        self.model = LRMF(n_users=self.n_users, n_entities=self.n_entities, l1=3, l2=3, kk=20)
        self.model.fit(R, candidates)

    def interview(self, answers: Dict[int, int], max_n_questions=5) -> List[int]:
        pass

    def predict(self, items: List[int], answers: Dict[int, int]) -> Dict[int, float]:
        pass

    def get_params(self) -> Dict:
        pass

    def load_params(self, params: Dict) -> None:
        pass


if __name__ == '__main__':
    training = pickle.load(open('../../partitioners/data/training.pkl', 'rb'))
    testing = pickle.load(open('../../partitioners/data/testing.pkl', 'rb'))
    meta = pickle.load(open('../../partitioners/data/meta.pkl', 'rb'))

    recommender = LRMFRecommender(meta)
    recommender.warmup(training)
