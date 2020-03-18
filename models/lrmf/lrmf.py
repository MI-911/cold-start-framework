from typing import List, Tuple, Union, Dict

import numpy as np
from scipy.linalg import solve_sylvester
from models.lrmf.maxvol import py_rect_maxvol
from tqdm import tqdm

LIKE = 1
DISLIKE = 0


def group_representation(users: List[int],
                         l1_questions: List[int], l2_questions: List[int],
                         ratings: np.ndarray) -> np.ndarray:
    ratings_submatrix = ratings[users, l1_questions + l2_questions]
    with_bias = np.hstack((ratings_submatrix, np.ones((len(users), 1))))
    return with_bias


def split_users(users: List[int], entity: int, ratings: np.ndarray) -> Tuple[List[int], List[int]]:
    user_ratings = ratings[users, entity]
    likes, = np.where(user_ratings == LIKE)
    dislikes, = np.where(user_ratings == DISLIKE)
    return likes, dislikes


def group_loss(users: List[int], entity_embeddings: np.ndarray, ratings: np.ndarray) -> float:
    _R = ratings[users]
    _B = group_representation(users, ratings)
    pass


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
    def __init__(self, l1_questions: List[int], depth: int, max_depth: int, lrmf: LRMF):
        self.depth = depth
        self.max_depth = max_depth
        self.l1_questions = l1_questions
        self.lrmf = lrmf

        self.users: List[int] = []
        self.question: Union[int, None] = None

    def is_leaf(self):
        return self.depth == self.max_depth

    def global_questions(self) -> List[int]:
        padding = [-1 for _ in range(self.max_depth - len(self.l1_questions))]
        return self.l1_questions + padding

    def local_questions(self, candidates) -> List[int]:
        questions, _ = py_rect_maxvol(self.lrmf.entity_embeddings[candidates], maxK=self.lrmf.l2)
        return questions

    def grow(self, users: List[int], candidates: List[int]):
        self.users = users

        min_loss, best_question = np.inf, None
        for candidate in candidates:
            likes, dislikes = split_users(users, candidate, self.lrmf.ratings)


    def interview_existing_user(self, user: int) -> np.ndarray:
        pass

    def interview_new_user(self, answers: Dict[int, int]) -> Union[int, np.ndarray]:
        pass