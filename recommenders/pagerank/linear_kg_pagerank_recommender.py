from typing import Dict, List

from recommenders.pagerank.linear_pagerank_recommender import LinearPageRankRecommender, GraphWrapper
from shared.meta import Meta
from shared.user import WarmStartUser


class LinearKGPageRankRecommender(LinearPageRankRecommender):
    def __init__(self, meta):
        super().__init__(meta)

    def construct_graph(self, training: Dict[int, WarmStartUser]) -> List[GraphWrapper]:
        # Get sentiments and entities
        sentiments = []
        for _, user in training.items():
            for entity, sentiment in user.training.items():
                sentiments.append(sentiment)

        # Create graphs
        sentiments = set(sentiments)
        graphs = []
        for sentiment in sentiments:
            graphs.append(GraphWrapper(training, sentiment, meta=self.meta, only_kg=True))

        return graphs
