import os
import pickle
from typing import Dict, List

from shared.meta import Meta
from shared.user import WarmStartUser, ColdStartUser


class DataLoader:
    def __init__(self, path):
        self.path = path

    def _load(self, file):
        with open(os.path.join(self.path, file), 'rb') as fp:
            return pickle.load(fp)

    def training(self) -> Dict[int, WarmStartUser]:
        return self._load('training.pkl')

    def testing(self) -> Dict[int, ColdStartUser]:
        return self._load('testing.pkl')

    def meta(self, training=None, recommendable_only: bool = False, type_limit: List = None) -> Meta:
        meta = self._load('meta.pkl')
        meta.recommendable_only = recommendable_only
        meta.type_limit = type_limit
        meta.popular_items = meta.get_question_candidates(training, recommendable_only=True) if training else None

        return meta
