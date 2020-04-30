import scipy
import networkx as nx


class SparseGraph:
    def __init__(self, graph: nx.Graph):
        self.node_list = graph.nodes()
        self.node_set = set(self.node_list)
        self.N = len(graph)
        self.M = nx.to_scipy_sparse_matrix(graph, nodelist=self.node_list, weight='weight', dtype=float)
        self.S = scipy.array(self.M.sum(axis=1)).flatten()
        self.S[self.S != 0] = 1.0 / self.S[self.S != 0]
        self.Q = scipy.sparse.spdiags(self.S.T, 0, *self.M.shape, format='csr')
        self.M = self.Q * self.M

    def scores(self, alpha=0.85, personalization=None, max_iterations=100, tol=1.0e-6):
        # Initialize with equal PageRank to each node
        x = scipy.repeat(1.0 / self.N, self.N)

        # Personalization vector
        if not personalization:
            p = scipy.repeat(1.0 / self.N, self.N)
        else:
            p = scipy.array([personalization.get(n, 0) for n in self.node_list], dtype=float)
            p = p / p.sum()

        # power iteration: make up to max_iter iterations
        for _ in range(max_iterations):
            last_x = x
            x = alpha * (x * self.M) + (1 - alpha) * p
            # check convergence, l1 norm
            err = scipy.absolute(x - last_x).sum()
            if err < self.N * tol:
                return dict(zip(self.node_list, map(float, x)))
        raise RuntimeError('PageRank failed to converge')
