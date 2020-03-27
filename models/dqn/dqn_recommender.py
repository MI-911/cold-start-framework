from typing import List, Dict

from models.shared.base_recommender import RecommenderBase
from models.shared.meta import Meta
from models.shared.user import WarmStartUser


class DqnRecommender(RecommenderBase):
    def __init__(self, meta: Meta, model: RecommenderBase):
        super(DqnRecommender, self).__init__(meta)

    def warmup(self, training: Dict[int, WarmStartUser], interview_length=5) -> None:
        pass

    def interview(self, answers: Dict, max_n_questions=5) -> List[int]:
        pass

    def predict(self, items: List[int], answers: Dict) -> Dict[int, float]:
        pass

    def get_parameters(self):
        pass

    def load_parameters(self, params):
        pass

