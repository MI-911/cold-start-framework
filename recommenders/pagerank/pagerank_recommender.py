from functools import reduce
from typing import List, Dict

from networkx import Graph, pagerank_scipy

from recommenders.base_recommender import RecommenderBase
from shared.meta import Meta
from shared.user import WarmStartUser

RATING_CATEGORIES = {1, 0, -1}


def construct_collaborative_graph(graph: Graph, training: Dict[int, WarmStartUser]):
    for user_id, user in training.items():
        user_id = f'user_{user_id}'
        graph.add_node(user_id, entity=False)

        for entity_idx, sentiment in user.training.items():
            graph.add_node(entity_idx, entity=True)
            graph.add_edge(user_id, entity_idx, sentiment=sentiment)

    return graph


def construct_knowledge_graph(meta: Meta):
    graph = Graph()

    for triple in meta.triples:
        head = meta.uri_idx[triple.head]
        tail = meta.uri_idx[triple.tail]

        graph.add_node(head, entity=True)
        graph.add_node(tail, entity=True)
        graph.add_edge(head, tail, type=triple.relation)

    return graph


class PageRankRecommender(RecommenderBase):
    def __init__(self, meta: Meta):
        super().__init__(meta)
        self.graph = None
        self.entity_indices = set()
        self.optimal_params = {
            'alpha': 0.85,
            'importance': {1: 0.9, 0: 0.1, -1: 0.0}
        }

    def construct_graph(self, training: Dict[int, WarmStartUser]):
        raise NotImplementedError()

    def _scores(self, alpha, node_weights, items):
        scores = pagerank_scipy(self.graph, alpha=alpha, personalization=node_weights).items()

        return {item: score for item, score in scores if item in items}

    @staticmethod
    def _weight(category, ratings, importance):
        if not ratings[category] or not importance[category]:
            return 0

        return importance[category] / len(ratings[category])

    def get_node_weights(self, answers, importance):
        ratings = {category: set() for category in RATING_CATEGORIES}

        for entity_idx, sentiment in answers.items():
            ratings[sentiment].add(entity_idx)

        # Find rated and unrated entities
        rated_entities = reduce(lambda a, b: a.union(b), ratings.values())
        unrated_entities = self.entity_indices.difference(rated_entities)

        # Treat unrated entities as unknown ratings
        ratings[0] = ratings[0].union(unrated_entities)

        # Compute the weight of each rating category
        rating_weight = {category: self._weight(category, ratings, importance) for category in RATING_CATEGORIES}

        # Assign weight to each node depending on their rating
        return {idx: rating_weight[category] for category in RATING_CATEGORIES for idx in ratings[category]}

    def fit(self, training: Dict[int, WarmStartUser]):
        for _, user in training.items():
            for entity in user.training.keys():
                self.entity_indices.add(entity)

        self.graph = self.construct_graph(training)

    def predict(self, items: List[int], answers: Dict[int, int]) -> Dict[int, float]:
        return self._scores(self.optimal_params['alpha'],
                            self.get_node_weights(answers, self.optimal_params['importance']), items)
