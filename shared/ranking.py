from random import shuffle
from typing import List, Dict
import itertools

from shared.enums import Sentiment


class Ranking:
    def __init__(self):
        self.sentiment_samples = dict()
        self._list = None
        self._data = None

    def set_data(self, data):
        self._data = data

    def get_data(self):
        # If no special data is used, return _list
        if self._data is None:
            # Ensure _list is set
            if self._list is None:
                self.to_list()

            return self._list
        else:
            return self._data

    def get_seen_samples(self):
        return set(itertools.chain.from_iterable(
            [value for key, value in self.sentiment_samples.items() if key != Sentiment.UNSEEN]))

    def to_list(self) -> List[int]:
        if self._list is None:
            as_list = list(itertools.chain.from_iterable(self.sentiment_samples.values()))

            # Shuffle to avoid any bias arising from
            shuffle(as_list)
            self._list = as_list

        return self._list

    def _get_utility(self, entity_idx, sentiment_utility: Dict[Sentiment, float]) -> float:
        for sentiment, utility in sentiment_utility.items():
            if entity_idx in self.sentiment_samples[sentiment]:
                return utility

        return 0

    def get_relevance(self, entity_indices: List[int]) -> List[bool]:
        return [entity_idx in self.sentiment_samples[Sentiment.POSITIVE] for entity_idx in entity_indices]

    def get_utility(self, entity_indices: List[int], sentiment_utility: Dict[Sentiment, float]) -> List[float]:
        return [self._get_utility(entity_idx, sentiment_utility) for entity_idx in entity_indices]
