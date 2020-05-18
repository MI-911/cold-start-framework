from typing import Dict, List

from recommenders.pagerank.linear_pagerank_recommender import LinearPageRankRecommender, GraphWrapper
from shared.user import WarmStartUser


class LinearCombinedPageRankRecommender(LinearPageRankRecommender):
    def clear_cache(self):
        for graph in self.graphs:
            graph.clear_cache()

    def __init__(self, meta, ask_limit: int = None):
        super().__init__(meta, ask_limit)

    def construct_graph(self, training: Dict[int, WarmStartUser]) -> List[GraphWrapper]:
        # Get sentiments and entities
        sentiments = set()
        for _, user in training.items():
            for entity, sentiment in user.training.items():
                sentiments.add(sentiment)

        # Create graphs for collaborative
        graphs = []
        for sentiment in sentiments:
            graphs.append(GraphWrapper(training, sentiment, self.meta, self.ask_limit))

        # Create graphs for kg
        for sentiment in sentiments:
            graphs.append(GraphWrapper(training, sentiment, self.meta, self.ask_limit, only_kg=True))

        return graphs
