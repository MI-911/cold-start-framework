from typing import Dict, List

from models.lrmf.lrmf import LRMF
from models.shared.base_recommender import RecommenderBase
from models.shared.user import WarmStartUser
import numpy as np


def get_rating_matrix(training, n_users, n_entities, rating_map=None):
    if not rating_map:
        rating_map = {
            1: 1,
            0: 0,
            -1: -1
        }

    R = np.zeros((n_users, n_entities))
    for user, data in training.items():
        for entity, rating in data['training'].items():
            R[user, entity] = rating_map[rating]

    return R

def extract_candidates(rating_matrix, n=100):
    n_ratings = rating_matrix.sum(axis=0)
    n_ratings = sorted([(entity, rs) for entity, rs in enumerate(n_ratings)])
    return [entity for entity, rs in n_ratings][:n]


class LRMFRecommender(RecommenderBase):
    def __init__(self, meta):
        super(LRMFRecommender, self).__init__()
        self.meta = meta
        self.model = None

        self.n_users = 0
        self.n_entities = 0

        self.n_candidates = 100

    def warmup(self, training: Dict) -> None:
        self.n_users = max(training.keys()) + 1
        self.n_entities = len(self.meta['entities'])

        self.model = LRMF(n_users=self.n_users, n_entities=self.n_entities, l1=3, l2=3, k=20)

        R = get_rating_matrix(training, self.n_users, self.n_entities, {
            1: 1,
            0: 0,
            -1: 0
        })
        candidates = extract_candidates(R, n=self.n_candidates)

        self.model.fit(R, candidates)

    def interview(self, answers: Dict[int, int], max_n_questions: int) -> List[int]:
        pass

    def predict(self, items: List[int], answers: Dict[int, int]) -> Dict[int, float]:
        pass

    def get_params(self) -> Dict:
        pass

    def load_params(self, params: Dict) -> None:
        pass

