from typing import Dict, List
from models.shared.base_recommender import RecommenderBase
from models.shared.user import WarmStartUser


class LRMFRecommender(RecommenderBase):
    def __init__(self, meta):
        super(LRMFRecommender, self).__init__()
        self.meta = meta

    def warmup(self, training: List[WarmStartUser]) -> None:
        pass

    def interview(self, answers: Dict[int, int], max_n_questions: int) -> List[int]:
        pass

    def predict(self, items: List[int], answers: Dict[int, int]) -> Dict[int, float]:
        pass

    def get_params(self) -> Dict:
        pass

    def load_params(self, params: Dict) -> None:
        pass

