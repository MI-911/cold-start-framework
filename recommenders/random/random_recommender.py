from random import randint
from typing import List, Dict

from recommenders.base_recommender import RecommenderBase
from shared.meta import Meta
from shared.user import WarmStartUser


class RandomRecommender(RecommenderBase):
    def __init__(self, meta: Meta):
        super().__init__(meta)

    def fit(self, training: Dict[int, WarmStartUser]):
        pass

    def predict(self, items: List[int], answers: Dict[int, int]) -> Dict[int, float]:
        return {item: randint(0, 1000) for item in items}
