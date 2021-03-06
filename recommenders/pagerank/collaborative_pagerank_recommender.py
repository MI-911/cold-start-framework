from typing import Dict

from networkx import Graph

from recommenders.pagerank.pagerank_recommender import PageRankRecommender, construct_collaborative_graph
from shared.user import WarmStartUser


class CollaborativePageRankRecommender(PageRankRecommender):
    def __init__(self, meta, ask_limit: int = None):
        super().__init__(meta, ask_limit)

    def construct_graph(self, training: Dict[int, WarmStartUser]):
        return construct_collaborative_graph(Graph(), training)
