import numpy as np


LIKE = 1
DISLIKE = 0


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
        self.V = np.zeros((self.kk, n_entities))     # Item embeddings

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
        # 2. Solve for user and transformation embeddings
        # 3. Solve for item embeddings
        pass

    def build_tree(self, training, height):
        # 1. For every item, determine how much it reduces loss.
        #    Pick the item with the largest loss decreases
        # 2. Split users into Like, Dislike, Unknown (explicit/implicit)
        # 3. Keep going until tree has a specific height
        pass

    def solve_T(self, users):
        # Solve the sylvester equation AX + XB = Q for X
        pass


class DecisionTree:
    def __init__(self, depth, users, parent_interview, LRMF):
        self.depth = depth
        self.users = users
        self.parent_interview = parent_interview
        self.LRMF = LRMF
        self.children = None
        self.candidate = None

    def grow(self, candidates):
        if len(self.parent_interview) >= self.depth:
            return
        if len(self.users) == 0:
            return

        # Find split item
        self.candidate = self.choose_candidate(candidates)
        u_l, u_d = self.split_users(self.candidate)
        interview = self.parent_interview + [self.candidate]

        # Send users and interview to children
        self.children = {
            LIKE: DecisionTree(self.depth, u_l, interview, self.LRMF),
            DISLIKE: DecisionTree(self.depth, u_d, interview, self.LRMF)
        }

        # Grow children recursively
        candidates.remove(self.candidate)
        self.children[LIKE].grow(candidates)
        self.children[DISLIKE].grow(candidates)

    def represent(self, users, questions):
        return self.LRMF.R[users][:, questions]

    def choose_candidate(self, candidates):
        # TODO: Implement
        return np.random.choice(candidates)

    def split_users(self, candidate):
        u_all = self.LRMF.R[self.users]
        u_l, = np.where(u_all[:, candidate] == LIKE)
        u_d, = np.where(u_all[:, candidate] == DISLIKE)

        return u_l, u_d

    def show(self):
        indent = ''.join([' |' for _ in range(len(self.parent_interview))])
        print(f'{indent} Node [q. {self.candidate}] ({len(self.users)} users)')
        if self.children:
            self.children[LIKE].show()
            self.children[DISLIKE].show()
