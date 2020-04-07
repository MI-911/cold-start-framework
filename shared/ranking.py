from typing import List, Dict
import itertools

from experiments.enums import Sentiment


class Ranking:
    def __init__(self):
        self.sentiment_samples = dict()

    def to_list(self) -> List[int]:
        return list(itertools.chain.from_iterable(self.sentiment_samples.values()))

    def get_utility(self, entity_idx, sentiment_utility: Dict[Sentiment, float]) -> float:
        for sentiment, utility in sentiment_utility.items():
            if entity_idx in self.sentiment_samples[sentiment]:
                return utility

        return 0
