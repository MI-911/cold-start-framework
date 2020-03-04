import os
import pickle
from typing import Dict

from models.shared.meta import Meta
from models.shared.user import WarmStartUser, ColdStartUser


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

    def meta(self) -> Meta:
        return self._load('meta.pkl')
