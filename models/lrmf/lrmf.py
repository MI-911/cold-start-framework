from typing import List, Tuple, Union, Dict

import numpy as np
from numpy.linalg import solve
from tqdm import tqdm

LIKE = 1
DISLIKE = 0


class LRMF:
    def __init__(self):
        pass

    def fit(self, ratings: np.ndarray, candidates: List[int]):
        pass

    def validate(self, user: int, to_validate: List[int]) -> Dict[int, float]:
        pass

    def interview(self, answers: Dict[int, int]) -> int:
        pass

    def rank(self, items: List[int], answers: Dict[int, int]):
        pass


class Tree:
    def __init__(self):
        pass

    def is_leaf(self):
        pass

    def grow(self, users: List[int], candidates: List[int]):
        pass

    def interview_existing_user(self, user: int) -> np.ndarray:
        pass

    def interview_new_user(self, answers: Dict[int, int]) -> Union[int, np.ndarray]:
        pass




