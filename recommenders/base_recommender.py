from typing import List, Dict

from shared.meta import Meta
from shared.user import WarmStartUser


class RecommenderBase:
    """ Base class that should be extended by every recommender model."""
    def __init__(self, meta: Meta):
        self.meta = meta

    def fit(self, training: Dict[int, WarmStartUser]):
        """
        Fits the model to the training data.
        """
        raise NotImplementedError

    def predict(self, items: List[int], answers: Dict[int, int]) -> Dict[int, float]:
        """
        Predicts a score for a list of items given a user's answers on items.
        """
        raise NotImplementedError
