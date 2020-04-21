from typing import List, Dict

from recommenders.base_recommender import RecommenderBase
from shared.meta import Meta
from shared.user import WarmStartUser


class DDPGRecommender(RecommenderBase):
    def __init__(self, meta: Meta):
        super(DDPGRecommender, self).__init__(meta)

    def fit(self, training: Dict[int, WarmStartUser]):
        # Train using reinforcement learning
        pass

    def predict(self, items: List[int], answers: Dict[int, int]) -> Dict[int, float]:
        # Rank the items given these answers
        pass
