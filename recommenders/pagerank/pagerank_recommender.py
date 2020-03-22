import pickle
from typing import List, Dict

from networkx import Graph

from recommenders.base_recommender import RecommenderBase
from shared.meta import Meta
from shared.user import WarmStartUser

from uuid import uuid4


def construct_collaborative_graph(graph: Graph, meta: Meta, training: List[WarmStartUser], only_positive=False):

    for user in training:
        user_id = f'user_{uuid4()}'
        graph.add_node(user_id, entity=False)

        for entity_idx, sentiment in user.training.items():
            graph.add_node(entity_idx, entity=True)
            graph.add_edge(user_id, entity_idx)

    for user, ratings in training:
        user_id = f'user_{user}'
        graph.add_node(user_id, entity=False)

        for rating in ratings:
            if only_positive and rating.rating != 1:
                continue

            graph.add_node(rating.e_idx, entity=True)
            graph.add_edge(user_id, rating.e_idx)

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

    def construct_graph(self, training: List[WarmStartUser]):
        raise NotImplementedError()

    def fit(self, training: List[WarmStartUser]):
        pass

    def predict(self, items: List[int], answers: Dict) -> Dict[int, float]:
        pass


if __name__ == '__main__':
    m = pickle.load(open('../../data/basic/split_0/meta.pkl', 'rb'))

    g = construct_knowledge_graph(m)
    pass
