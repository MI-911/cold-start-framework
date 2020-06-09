from typing import Dict, List

from recommenders.pagerank.linear_pagerank_recommender import GraphWrapper
from recommenders.pagerank.pair_linear_pagerank_recommender import PairLinearPageRankRecommender
from shared.user import WarmStartUser


class PairLinearJointPageRankRecommender(PairLinearPageRankRecommender):
    def __init__(self, meta, ask_limit: int = None):
        super().__init__(meta, ask_limit)

    def construct_graph(self, training: Dict[int, WarmStartUser]) -> List[GraphWrapper]:
        # Get sentiments and entities
        sentiments = set()
        for _, user in training.items():
            for entity, sentiment in user.training.items():
                sentiments.add(sentiment)

        sentiments = sorted(list(sentiments), reverse=True)  # Ensure order, like, dont know, dislike.

        # Create graphs for collaborative
        graphs = []
        for sentiment in sentiments:
            graphs.append(GraphWrapper(training, sentiment, self.meta, self.ask_limit))

        # Create graphs for kg
        for sentiment in sentiments:
            graphs.append(GraphWrapper(training, sentiment, self.meta, self.ask_limit, only_kg=True))

        # Create graphs for joint
        for sentiment in sentiments:
            graphs.append(GraphWrapper(training, sentiment, self.meta, self.ask_limit, use_meta=True))

        return graphs
