import numpy as np
from numpy.linalg import inv, norm
from scipy.linalg import solve_sylvester
from typing import List
from loguru import logger
from tqdm import tqdm

LIKE = 1
DISLIKE = 0


class DecisionTree:
    def __init__(self, depth, users, parent_interview, LRMF):
        self.depth = depth
        self.users = users
        self.parent_interview = parent_interview
        self.LRMF = LRMF
        self.children = None
        self.candidate = None

        self.T = self.solve_transformation(users, parent_interview)

    def is_leaf(self):
        return self.depth == len(self.parent_interview)

    def has_children(self):
        return self.children and any(self.children.values())

    def grow(self, candidates):
        if self.is_leaf():
            return

        # Find split item
        self.candidate = self.choose_candidate(candidates)
        u_l, u_d = self.split_users(self.candidate)
        interview = self.parent_interview + [self.candidate]

        # Send users and interview to children, only construct a child
        # if there are users to allocate to it
        self.children = {
            LIKE: None if not any(u_l) else DecisionTree(self.depth, u_l, interview, self.LRMF),
            DISLIKE: None if not any(u_d) else DecisionTree(self.depth, u_d, interview, self.LRMF)
        }

        # Grow children recursively
        [child.grow([c for c in candidates if not c == self.candidate])
         for child in self.children.values()
         if child is not None]

    def solve_transformation(self, users, questions):
        alpha = self.LRMF.alpha
        V = self.LRMF.V
        R = self.LRMF.R[users]
        B = self.represent(users, questions)

        # Solve the sylvester equation AX + XB = Q
        VVT = inv(V @ V.T)

        _A = B.T @ B
        _B = alpha * VVT
        _Q = B.T @ R @ V.T @ VVT

        return solve_sylvester(_A, _B, _Q)

    def represent(self, users, questions):
        # Returns a matrix representation of group Ug ; e
        base = np.ones((len(users), len(questions) + 1))
        user_rep = self.LRMF.R[users][:, questions]
        base[:, :-1] = user_rep  # Leave the last column all 1s
        return base

    def interview(self, user_id: int, user_ratings: List[int]):
        # User should be a row from R
        def _transform_user(id):
            return self.represent([id], self.parent_interview) @ self.T

        if self.is_leaf():
            return _transform_user(user_id)

        answer = user_ratings[self.candidate]
        next = self.children[answer]

        if next is None:
            return _transform_user(user_id)
        return next.interview(user_id, user_ratings)

    def transform_leaves(self):
        # NOTE: DEPRECATED
        if not self.is_leaf() and self.has_children():
            return np.vstack((
                child.transform_leaves() for rating, child in self.children.items()
                if child is not None
            ))
        return self.represent(self.users, self.parent_interview) @ self.T

    def choose_candidate_test(self, candidates):
        return np.random.choice(candidates)

    def choose_candidate(self, candidates):
        candidate_losses = {}

        for candidate in tqdm(candidates, desc=f"[Choosing split item at depth {len(self.parent_interview)}]"):
            u_l, u_d = self.split_users(candidate)
            loss = 0
            for us in [u_l, u_d]:
                questions = self.parent_interview + [candidate]
                T = self.solve_transformation(us, questions)
                B = self.represent(us, questions)
                R = self.LRMF.R[us]
                V = self.LRMF.V
                alpha = self.LRMF.alpha

                loss += norm(R - B @ T @ V) + alpha * norm(T)

            candidate_losses[candidate] = loss

        sorted_candidates = [candidate for candidate, loss in sorted(candidate_losses.items(), key=lambda x: x[1])]
        return sorted_candidates[0]

    def split_users(self, candidate):
        u_all = self.LRMF.R[self.users]
        u_l, = np.where(u_all[:, candidate] == LIKE)
        u_d, = np.where(u_all[:, candidate] == DISLIKE)

        return u_l, u_d

    def show(self):
        indent = ''.join([' |' for _ in range(len(self.parent_interview))])
        print(f'{indent} Node [q. {self.candidate}] ({len(self.users)} users)')
        if self.children:
            [child.show() for child in self.children.values() if child is not None]
