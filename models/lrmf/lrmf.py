from typing import List, Tuple, Union, Dict

import numpy as np
from numpy.linalg import solve
from tqdm import tqdm

LIKE = 1
DISLIKE = 0


class LRMF:
    def __init__(self, n_users, n_entities, l1, l2, kk, regularisation):
        """
        Instantiates an LRMF model for conducting interviews and making
        recommendations. The model conducts an interview of total length L = l1 + l2.

        :param n_users: The number of users.
        :param n_entities: The number of entities.
        :param l1: The number of global questions to be used for group division.
        :param l2: The number of local questions to be asked in every group.
        :param kk: The number of latent factors for entity embeddings.
        :param regularisation: Control parameter for l2-norm regularisation.
        """
        self.n_users = n_users
        self.n_entities = n_entities
        self.l1 = l1
        self.l2 = l2
        self.kk = kk
        self.regularisation = regularisation

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




