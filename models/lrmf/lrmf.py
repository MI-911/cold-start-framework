import numpy as np


class LRMF:
    def __init__(self, l1, l2, n_users, n_entities, k):
        self.n_users = n_users
        self.n_entities = n_entities
        self.candidates = None

        self.l1 = l1                                # Num. global questions
        self.l2 = l2                                # Num. local questions

        self.l = l1 + l2 + 1                        # Question dimension
        self.k = k                                  # Embedding dimension

        self.R = np.zeros((n_users, n_entities))    # Ratings matrix
        self.T = np.zeros((self.l, k))              # Transformation matrix
        self.V = np.zeros((self.l, n_entities))     # Item embeddings

        self.l1_questions = [0 for _ in range(l1)]  # Indices of global questions
        self.l2_questions = [0 for _ in range(l2)]  # Indices of local questions

    def fit(self, training, candidates):
        # 1. Build the tree
        self.build_tree(training, 5)
        # 2. Solve for user and transformation embeddings
        # 3. Solve for item embeddings
        pass

    def build_tree(self, training, height):
        # 1. For every item, determine how much it reduces loss.
        #    Pick the item with the largest loss decreases
        # 2. Split users into Like, Dislike, Unknown (explicit/implicit)
        # 3. Keep going until tree has a specific height
        pass