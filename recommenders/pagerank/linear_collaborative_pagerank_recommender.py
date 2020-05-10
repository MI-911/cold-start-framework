from typing import Dict, List

from recommenders.pagerank.linear_pagerank_recommender import LinearPageRankRecommender, GraphWrapper
from shared.user import WarmStartUser


class LinearCollaborativePageRankRecommender(LinearPageRankRecommender):
    def __init__(self, meta):
        super().__init__(meta)

    def construct_graph(self, training: Dict[int, WarmStartUser]) -> List[GraphWrapper]:
        # Get sentiments and entities
        sentiments = set()
        for _, user in training.items():
            for entity, sentiment in user.training.items():
                sentiments.add(sentiment)

        # Create graphs
        graphs = []
        for sentiment in sentiments:
            graphs.append(GraphWrapper(training, sentiment))

        return graphs
