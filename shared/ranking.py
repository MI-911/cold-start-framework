from typing import List
import itertools


class Ranking:
    def __init__(self):
        self.sentiment_samples = dict()

    def to_list(self) -> List[int]:
        return list(itertools.chain.from_iterable(self.sentiment_samples.values()))
