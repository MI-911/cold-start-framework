from typing import List, Tuple, Union, Dict

import numpy as np
from numpy.linalg import solve
from tqdm import tqdm

LIKE = 5
UNKNOWN = 0
DISLIKE = 1


def split_users(users: List[int], entity: int, ratings: np.ndarray) -> Tuple[List[int], List[int], List[int]]:
    entity_ratings = ratings[users, entity]
    likes, = np.where(entity_ratings == LIKE)
    dislikes, = np.where(entity_ratings == DISLIKE)
    unknowns, = np.where(entity_ratings == UNKNOWN)
    return likes, dislikes, unknowns


def optimal_entity_embeddings(ratings: np.ndarray, tree, n_latent_factors: int, regularization: float) -> np.ndarray:
    n_users, n_entities = ratings.shape
    entity_embeddings = np.zeros((n_entities, n_latent_factors))

    for e in tqdm(range(n_entities), desc=f'[Optimizing entity embeddings]'):
        A = np.zeros((n_latent_factors, n_latent_factors))
        B = np.zeros((1, n_latent_factors))

        rating_users, = np.where(ratings[:, e] != UNKNOWN)
        for u in rating_users:
            r = ratings[u, e]
            _Ta = tree.interview_existing_user(u)

            A += _Ta.T @ _Ta + np.eye(n_latent_factors) * regularization
            B += r * _Ta

        try:
            entity_embeddings[e] = solve(A, B.reshape(-1))
        except np.linalg.LinAlgError:
            # A is a singular matrix, nothing to solve
            entity_embeddings[e] = np.zeros(n_latent_factors)

    return entity_embeddings


def optimal_group_embedding(users: List[int], entity_embeddings: np.ndarray, ratings: np.ndarray,
                            regularization: float) -> np.ndarray:
    A = np.zeros((entity_embeddings.shape[1], entity_embeddings.shape[1]))
    B = np.zeros((1, entity_embeddings.shape[1]))

    for user in users:
        rated_entities, = np.where(ratings[user] != UNKNOWN)
        for entity in rated_entities:
            rating = ratings[user, entity]
            v = entity_embeddings[entity]
            A += v.T @ v + np.eye(entity_embeddings.shape[1]) * regularization
            B += rating * v
    try:
        return solve(A, B.reshape(-1)).reshape(1, -1)
    except np.linalg.LinAlgError:
        # A is a singular matrix, nothing we can do
        return np.zeros((1, entity_embeddings.shape[1]))


def group_loss(users_in_group: List[int], group_embedding: np.ndarray,
               entity_embeddings: np.ndarray, ratings: np.ndarray) -> float:
    ratings_submatrix = ratings[users_in_group]
    user_embeddings = np.ones((len(users_in_group), 1)) * group_embedding
    predicted_ratings_submatrix = user_embeddings @ entity_embeddings.T
    return ((ratings_submatrix - predicted_ratings_submatrix) ** 2).sum()


class FMF:
    def __init__(self, n_users, n_entities, max_depth: int, n_latent_factors: int, regularization: float):
        """
        Functional Matrix Factorisation model for conducting interviews
        making recommendations.
        @param n_users: Number of users
        @param n_entities: Number of entities
        @param max_depth: The maximum length of the interview
        @param n_latent_factors: The number of latent factors for embedding users and entities
        @param regularization: Regularisation used for both users and entities.
        """
        self.n_users = n_users
        self.n_entities = n_entities

        self.max_depth = max_depth
        self.n_latent_factors = n_latent_factors
        self.regularization = regularization

        self.ratings = np.zeros((n_users, n_entities))
        self.entity_embeddings: np.ndarray = np.random.rand(n_entities, n_latent_factors)
        self.T: Tree = Tree(depth=0, max_depth=max_depth, fmf=self)

    def fit(self, ratings: np.ndarray, candidates: List[int]):
        self.ratings = ratings

        all_users = [u for u in range(self.n_users)]

        # self.T.grow_test(all_users, candidates)
        self.T.grow(all_users, candidates)
        self.entity_embeddings = optimal_entity_embeddings(ratings, self.T,
                                                           self.n_latent_factors, self.regularization)

    def validate(self, user: int, to_validate: List[int]) -> Dict[int, float]:
        u = self.T.interview_existing_user(user)
        similarities, = u @ self.entity_embeddings[to_validate].T
        return {e: s for e, s in zip(to_validate, similarities)}

    def interview(self, answers: Dict[int, int]) -> int:
        return self.T.interview_new_user(answers)

    def rank(self, items: List[int], answers: Dict[int, int]):
        u = self.T.interview_new_user(answers)
        if not type(u) == np.ndarray:
            raise ValueError(
                f"FMF: Expected to receive a user embedding from the interview, but " +
                f"received a question instead. Interviewed with {len(answers)} answers " +
                f"on a tree of height {self.T.max_depth}."
            )

        similarities, = u @ self.entity_embeddings[items].T
        return {e: s for e, s in zip(items, similarities)}


class Tree:
    def __init__(self, depth: int, max_depth: int, fmf: FMF):
        self.depth = depth
        self.max_depth = max_depth
        self.fmf = fmf
        self.user_embedding: np.ndarray = np.random.rand(1, fmf.n_latent_factors)
        self.entity_embeddings: np.ndarray = fmf.entity_embeddings

        self.users: List[int] = []
        self.question: Union[int, None] = None
        # Continue tree growth downwards
        self.children = None if self.is_leaf() else {
            LIKE: Tree(depth=depth+1, max_depth=max_depth, fmf=fmf),
            DISLIKE: Tree(depth=depth+1, max_depth=max_depth, fmf=fmf),
            UNKNOWN: Tree(depth=depth+1, max_depth=max_depth, fmf=fmf)
        }

    def is_leaf(self):
        return self.depth == self.max_depth

    def grow_test(self, users: List[int], candidates: List[int]):
        self.users = users
        if self.is_leaf():
            self.user_embedding = optimal_group_embedding(self.users, self.fmf.entity_embeddings, self.fmf.ratings,
                                                          self.fmf.regularization)
            return

        self.question = np.random.choice(candidates)
        likes, dislikes, unknowns = split_users(self.users, self.question, self.fmf.ratings)

        self.children[LIKE].grow_test(likes, [c for c in candidates if not c == self.question])
        self.children[DISLIKE].grow_test(dislikes, [c for c in candidates if not c == self.question])
        self.children[UNKNOWN].grow_test(unknowns, [c for c in candidates if not c == self.question])

    def grow(self, users: List[int], candidates: List[int]):
        self.users = users

        if self.is_leaf():
            self.user_embedding = optimal_group_embedding(self.users, self.fmf.entity_embeddings, self.fmf.ratings,
                                                          self.fmf.regularization)
            return

        min_loss, question = np.inf, None

        for candidate in tqdm(candidates, desc=f'[Searching candidates at depth {self.depth}]'):
            # Split users on this candidate
            likes, dislikes, unknowns = split_users(users, candidate, self.fmf.ratings)
            # Get optimal profiles for both groups
            uL = optimal_group_embedding(likes, self.fmf.entity_embeddings, self.fmf.ratings, self.fmf.regularization)
            uD = optimal_group_embedding(dislikes, self.fmf.entity_embeddings, self.fmf.ratings, self.fmf.regularization)
            uU = optimal_group_embedding(unknowns, self.fmf.entity_embeddings, self.fmf.ratings, self.fmf.regularization)
            # Get the prediction loss for both groups
            loss = group_loss(likes, uL, self.fmf.entity_embeddings, self.fmf.ratings)
            loss += group_loss(dislikes, uD, self.fmf.entity_embeddings, self.fmf.ratings)
            loss += group_loss(unknowns, uU, self.fmf.entity_embeddings, self.fmf.ratings)

            if loss < min_loss:
                question = candidate

        self.question = question
        likes, dislikes, unknowns = split_users(self.users, self.question, self.fmf.ratings)

        self.children[LIKE].grow(likes, [c for c in candidates if not c == self.question])
        self.children[DISLIKE].grow(dislikes, [c for c in candidates if not c == self.question])
        self.children[UNKNOWN].grow(unknowns, [c for c in candidates if not c == self.question])

    def interview_existing_user(self, user: int) -> np.ndarray:
        if self.is_leaf():
            return self.user_embedding
        return self.children[self.fmf.ratings[user, self.question]].interview_existing_user(user)

    def interview_new_user(self, answers: Dict[int, int]) -> Union[int, np.ndarray]:
        if self.is_leaf():
            return self.user_embedding
        if not answers:
            return self.question
        answer = answers[self.question] if self.question in answers else UNKNOWN
        return self.children[answer].interview_new_user({e: a for e, a in answers.items() if not e == self.question})




