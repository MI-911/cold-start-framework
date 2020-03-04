import os
import pickle
from typing import Dict

from models.shared.meta import Meta
from models.shared.user import WarmStartUser, ColdStartUser


class DataLoader:
    def __init__(self, path):
        self.path = path

    def load_training(self) -> Dict[int, WarmStartUser]:
        with open(os.path.join(self.path, 'training.pkl'), 'rb') as fp:
            return pickle.load(fp)

    def load_testing(self) -> Dict[int, ColdStartUser]:
        with open(os.path.join(self.path, 'testing.pkl'), 'rb') as fp:
            return pickle.load(fp)

    def load_meta(self) -> Meta:
        with open(os.path.join(self.path, 'meta.pkl'), 'rb') as fp:
            return pickle.load(fp)
