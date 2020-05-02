from typing import Dict

from recommenders.pagerank.pagerank_recommender import PageRankRecommender, construct_collaborative_graph, \
    construct_knowledge_graph
from shared.user import WarmStartUser


class JointPageRankRecommender(PageRankRecommender):
    def __init__(self, meta, ask_limit: int = None):
        super().__init__(meta, ask_limit)

    def construct_graph(self, training: Dict[int, WarmStartUser]):
        return construct_collaborative_graph(construct_knowledge_graph(self.meta), training)
