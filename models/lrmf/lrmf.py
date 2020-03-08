import numpy as np
from models.lrmf.decision_tree import DecisionTree
from numpy.linalg import inv
from loguru import logger
from tqdm import tqdm


class LRMF:
    def __init__(self, l1, l2, n_users, n_entities, kk, alpha, beta):
        self.n_users = n_users
        self.n_entities = n_entities
        self.candidates = None

        self.l1 = l1                                # Num. global questions
        self.l2 = l2                                # Num. local questions

        self.k = l1 + l2 + 1                         # Question dimension
        self.kk = kk                                 # Embedding dimension

        self.R = np.zeros((n_users, n_entities))         # Ratings matrix
        self.T = np.zeros((self.k, kk))                  # Transformation matrix
        self.V = np.random.rand(self.kk, n_entities)     # Item embeddings

        # Regularisation
        self.alpha = alpha
        self.beta = beta

        self.tree = None

    def fit(self, R, candidates):
        self.R = R

        # 1. Build the tree, optimise transformations
        logger.info(f'Building tree...')
        self.tree = DecisionTree(
            depth=self.l1 + self.l2,
            users=[u for u in range(self.n_users)],
            parent_interview=[],
            LRMF=self)
        self.tree.grow(candidates)

        # 2. Optimise item embeddings
        logger.info('Optimising item embeddings...')
        self.V = self.solve_item_embeddings()

    def rank(self, items, answers):
        u = self.tree.get_interview_representation(answers, [])
        similarities = u @ self.V[items]
        return {entity_id: similarity for entity_id, similarity in zip(items, similarities)}

    def validate(self, user_id, items):
        u = self.tree.get_representation(user_id, self.R[user_id])
        similarities, = u @ self.V[:, items]
        return {item: similarity for item, similarity in zip(items, similarities)}

    def interview(self, answers):
        return self.tree.get_next_question(answers)

    def solve_item_embeddings(self):
        S = np.zeros((self.n_users, self.kk))

        for u in tqdm(range(self.n_users), desc='[Optimizing item embeddings]'):
            S[u] = self.tree.get_representation(u, self.R[u])

        # Ordinary least squares solution
        # NOTE: Paper says I_l, but it should be I_k'
        return inv(S.T @ S + self.beta * np.eye(self.kk)) @ S.T @ self.R
