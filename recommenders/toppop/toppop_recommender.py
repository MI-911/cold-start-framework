from typing import List, Dict

from loguru import logger

from recommenders.base_recommender import RecommenderBase
from shared.meta import Meta
from shared.user import WarmStartUser


class TopPopRecommender(RecommenderBase):
    def __init__(self, meta: Meta, likes_only=False):
        super().__init__(meta)

        self.entity_ratings = dict()
        self.likes_only = likes_only

    def fit(self, training: Dict[int, WarmStartUser]):
        for idx, user in training.items():
            for entity, rating in user.training.items():
                # Skip don't knows
                if rating == 0:
                    continue

                if self.likes_only and rating != 1:
                    # logger.info(f'Skip {rating}')
                    continue

                self.entity_ratings[entity] = self.entity_ratings.get(entity, 0) + 1

    def predict(self, items: List[int], answers: Dict[int, int]) -> Dict[int, float]:
        return {item: self.entity_ratings.get(item, 0) for item in items}
