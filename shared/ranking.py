from typing import List, Dict
import itertools

from experiments.enums import Sentiment


class Ranking:
    def __init__(self):
        self.sentiment_samples = dict()

    def to_list(self) -> List[int]:
        return list(itertools.chain.from_iterable(self.sentiment_samples.values()))

    def _get_utility(self, entity_idx, sentiment_utility: Dict[Sentiment, float]) -> float:
        for sentiment, utility in sentiment_utility.items():
            if entity_idx in self.sentiment_samples[sentiment]:
                return utility

        return 0

    def get_relevance(self, entity_indices: List[int]) -> List[bool]:
        return [entity_idx in self.sentiment_samples[Sentiment.POSITIVE] for entity_idx in entity_indices]

    def get_utility(self, entity_indices: List[int], sentiment_utility: Dict[Sentiment, float]) -> List[float]:
        return [self._get_utility(entity_idx, sentiment_utility) for entity_idx in entity_indices]
