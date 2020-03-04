import numpy as np
from models.lrmf.decision_tree import DecisionTree


class LRMF:
    def __init__(self, l1, l2, n_users, n_entities, kk):
        self.n_users = n_users
        self.n_entities = n_entities
        self.candidates = None

        self.l1 = l1                                # Num. global questions
        self.l2 = l2                                # Num. local questions

        self.k = l1 + l2 + 1                         # Question dimension
        self.kk = kk                                 # Embedding dimension

        self.R = np.zeros((n_users, n_entities))     # Ratings matrix
        self.T = np.zeros((self.k, kk))              # Transformation matrix
        self.V = np.random.rand(self.kk, n_entities)     # Item embeddings

        # Regularisation
        self.alpha = 0.001
        self.beta = 0.001

    def fit(self, R, candidates):
        self.R = R
        # 1. Build the tree
        tree = DecisionTree(
            depth=self.l1 + self.l2,
            users=[u for u in range(self.n_users)],
            parent_interview=[],
            LRMF=self)
        tree.grow(candidates)
        tree.show()

        pass

    def solve_T(self, users):
        # Solve the sylvester equation AX + XB = Q for X
        pass