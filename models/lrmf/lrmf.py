from typing import List, Tuple, Union, Dict

import numpy as np
from loguru import logger
from scipy.linalg import solve_sylvester, inv, LinAlgError
from models.lrmf.maxvol import py_rect_maxvol
from tqdm import tqdm

LIKE = 5
DISLIKE = 1


def except_item(lst, item):
    return [i for i in lst if not i == item]


def group_representation(users: List[int],
                         l1_questions: List[int], l2_questions: List[int],
                         ratings: np.ndarray) -> np.ndarray:
    ratings_submatrix = ratings[users][:, l1_questions + l2_questions]
    with_bias = np.hstack((ratings_submatrix, np.ones((len(users), 1))))
    return with_bias


def split_users(users: List[int], entity: int, ratings: np.ndarray) -> Tuple[List[int], List[int]]:
    user_ratings = ratings[users, entity]
    likes, = np.where(user_ratings == LIKE)
    dislikes, = np.where(user_ratings == DISLIKE)
    return likes, dislikes


def global_questions_vector(questions: List[int], max_length: int) -> List[int]:
    padding = [-1 for _ in range(max_length - len(questions))]
    return questions + padding


def local_questions_vector(candidates: List[int], entity_embeddings: np.ndarray, max_length: int) -> List[int]:
    questions, _ = py_rect_maxvol(entity_embeddings[candidates], maxK=max_length)
    return questions.tolist()


def group_loss(users: List[int], global_questions: List[int], local_questions: List[int],
               entity_embeddings: np.ndarray, ratings: np.ndarray) -> float:
    _B = group_representation(users, global_questions, local_questions, ratings)
    _T = transformation(users, _B, entity_embeddings, ratings)

    group_ratings = ratings[users]
    predicted_ratings = _B @ _T @ entity_embeddings.T
    loss = ((group_ratings - predicted_ratings) ** 2).sum()
    regularisation = _T.sum() ** 2

    return loss + regularisation


def transformation(users: List[int],
                   representation: np.ndarray, entity_embeddings: np.ndarray,
                   ratings: np.ndarray, alpha=1.0) -> np.ndarray:
    try:
        _A = representation.T @ representation
        _B = alpha * inv(entity_embeddings.T @ entity_embeddings)
        _Q = representation.T @ ratings[users] @ entity_embeddings @ inv(entity_embeddings.T @ entity_embeddings)
        return solve_sylvester(_A, _B, _Q)
    except LinAlgError as err:
        return np.zeros((representation.shape[1], entity_embeddings.shape[1]))


def optimise_entity_embeddings(ratings: np.ndarray, tree, k: int, kk: int, regularisation: float) -> np.ndarray:
    n_users, n_entities = ratings.shape
    _S = np.zeros((n_users, kk))

    for u in tqdm(range(n_users), desc="[Optimizing entity embeddings...]"):
        _S[u] = tree.interview_existing_user(u)

    try:
        _A = inv(_S.T @ _S) + np.eye(kk) * regularisation
        _B = _S.T @ ratings
        return (_A @ _B).T
    except LinAlgError as err:
        return np.zeros((n_entities, kk))


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

        self.interview_length: int = self.l1 + self.l2
        self.k: int = self.interview_length + 1

        self.ratings: np.ndarray = np.zeros((self.n_users, self.n_entities))
        self.entity_embeddings: np.ndarray = np.random.rand(self.n_entities, self.kk)

        self.T = Tree(l1_questions=[], depth=0, max_depth=self.l1, lrmf=self)

    def fit(self, ratings: np.ndarray, candidates: List[int]):
        self.ratings = ratings

        all_users = [u for u in range(self.n_users)]

        self.T.grow(all_users, candidates)
        self.entity_embeddings = optimise_entity_embeddings(ratings, self.T, self.k, self.kk, self.regularisation)

    def validate(self, user: int, to_validate: List[int]) -> Dict[int, float]:
        user_vector = self.T.interview_existing_user(user)
        similarities = user_vector @ self.entity_embeddings[to_validate].T
        return {e: s for e, s in zip(to_validate, similarities)}

    def interview(self, answers: Dict[int, int]) -> int:
        return self.T.interview_new_user(answers, {})

    def rank(self, items: List[int], answers: Dict[int, int]):
        user_vector = self.T.interview_new_user(answers, {})
        similarities = user_vector @ self.entity_embeddings[items].T
        return {e: s for e, s in zip(items, similarities)}


class Tree:
    def __init__(self, l1_questions: List[int], depth: int, max_depth: int, lrmf: LRMF):
        self.depth = depth
        self.max_depth = max_depth
        self.l1_questions = l1_questions
        self.l2_questions: List[int] = []
        self.lrmf = lrmf

        self.users: List[int] = []
        self.transformation: Union[np.ndarray, None] = None
        self.question: Union[int, None] = None

        self.children: Union[Dict[int, Tree], None] = None

    def is_leaf(self):
        return self.depth == self.max_depth

    def grow(self, users: List[int], candidates: List[int]):
        self.users = users
        self.l2_questions = local_questions_vector(
            candidates, self.lrmf.entity_embeddings, self.lrmf.l2)

        if self.is_leaf():
            self.transformation = transformation(
                self.users, group_representation(self.users, self.l1_questions, self.l2_questions, self.lrmf.ratings),
                self.lrmf.entity_embeddings, self.lrmf.ratings)

            return

        min_loss, best_question = np.inf, None
        for candidate in tqdm(candidates, desc=f'[Selecting question at depth {self.depth} ]'):
            likes, dislikes = split_users(users, candidate, self.lrmf.ratings)

            loss = 0
            for group in [likes, dislikes]:
                rest_candidates = except_item(candidates, candidate)
                global_questions = global_questions_vector(self.l1_questions, self.lrmf.l1)
                local_questions = local_questions_vector(rest_candidates, self.lrmf.entity_embeddings, self.lrmf.l2)
                loss += group_loss(
                    group, global_questions, local_questions,self.lrmf.entity_embeddings, self.lrmf.ratings)

            if loss < min_loss:
                min_loss = loss
                best_question = candidate

        self.question = best_question
        remaining_candidates = except_item(candidates, self.question)
        self.l2_questions = local_questions_vector(
            remaining_candidates, self.lrmf.entity_embeddings, self.lrmf.l2)

        self.transformation = transformation(
            self.users, group_representation(self.users, self.l1_questions, self.l2_questions, self.lrmf.ratings),
            self.lrmf.entity_embeddings, self.lrmf.ratings)

        self.children = {
            LIKE: Tree(self.l1_questions + [self.question], self.depth + 1, self.max_depth, self.lrmf),
            DISLIKE: Tree(self.l1_questions + [self.question], self.depth + 1, self.max_depth, self.lrmf)
        }

        likes, dislikes = split_users(self.users, self.question, self.lrmf.ratings)
        self.children[LIKE].grow(likes, remaining_candidates)
        self.children[DISLIKE].grow(dislikes, remaining_candidates)

    def interview_existing_user(self, user: int) -> np.ndarray:
        if self.is_leaf():
            user_vector, = group_representation([user], self.l1_questions, self.l2_questions, self.lrmf.ratings)
            return user_vector @ self.transformation

        answer = self.lrmf.ratings[user, self.question]
        return self.children[answer].interview_existing_user(user)

    def interview_new_user(self, answers: Dict[int, int], user_answers: Dict[int, int]) -> Union[int, np.ndarray]:
        if self.is_leaf():
            # Have we asked all our local questions?
            if len(user_answers) < self.lrmf.interview_length:
                for local_question in self.l2_questions:
                    # First try to exhaust the available answers
                    if local_question in answers:
                        answer = answers[local_question]
                        user_answers[local_question] = answer

                        return self.interview_new_user({
                            question: answer
                            for question, answer in answers.items()
                            if not question == local_question
                        }, user_answers)

                    # If we cannot get an answer from the arguments, return the question
                    elif local_question not in user_answers:
                        return local_question

            # If we have asked all of our questions, return the transformed user vector
            else:
                user_vector = [a for e, a in user_answers.items()]
                user_vector.append(1)  # Add bias
                return user_vector @ self.transformation

        if not answers:
            return self.question

        answer = answers[self.question] if self.question in answers else DISLIKE
        user_answers[self.question] = answer

        return self.children[answer].interview_new_user({
                question: answer
                for question, answer
                in answers.items() if not question == self.question
            }, user_answers
        )
