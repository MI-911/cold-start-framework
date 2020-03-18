from typing import List, Tuple, Union, Dict

import numpy as np
from numpy.linalg import solve
from tqdm import tqdm

LIKE = 1
DISLIKE = 0


class LRMF:
    def __init__(self, n_users: int, n_entities: int, l1: int, l2: int, kk: int, regularisation: float):
        """
        Instantiates an LRMF model for conducting interviews and making
        recommendations. The model conducts an interview of total length L = l1 + l2.

        :param n_users: The number of users.
        :param n_entities: The number of entities.
        :param l1: The number of global questions to be used for group division.
        :param l2: The number of local questions to be asked in every group.
        :param kk: The number of latent factors for entity embeddings.
                   NOTE: Due to a seemingly self-imposed restriction from the paper, the number
                   of latent factors used to represent entities must be exactly l2, and cannot be kk.
                   We have emailed the authors requesting an explanation and are awaiting a response.
        :param regularisation: Control parameter for l2-norm regularisation.
        """
        self.n_users = n_users
        self.n_entities = n_entities
        self.l1 = l1
        self.l2 = l2
        self.kk = l2  # See the note
        self.regularisation = regularisation

        self.interview_length: int = l1 + l2
        self.k: int = self.interview_length + 1

        self.ratings: np.ndarray = np.zeros((n_users, n_entities))
        self.entity_embeddings: np.ndarray = np.zeros((n_entities, kk))

    def fit(self, ratings: np.ndarray, candidates: List[int]):
        pass

    def validate(self, user: int, to_validate: List[int]) -> Dict[int, float]:
        pass

    def interview(self, answers: Dict[int, int]) -> int:
        pass

    def rank(self, items: List[int], answers: Dict[int, int]):
        pass


class Tree:
    def __init__(self, depth: int, max_depth: int, lrmf: LRMF):
        self.depth = depth
        self.max_depth = max_depth
        self.lrmf = lrmf

    def is_leaf(self):
        return self.depth == self.max_depth

    def grow(self, users: List[int], candidates: List[int]):
        pass

    def interview_existing_user(self, user: int) -> np.ndarray:
        pass

    def interview_new_user(self, answers: Dict[int, int]) -> Union[int, np.ndarray]:
        pass