from os import environ
from random import randint
from typing import List, Dict

from neo4j import GraphDatabase

from recommenders.base_recommender import RecommenderBase
from shared.meta import Meta
from shared.user import WarmStartUser

_uri = environ.get('BOLT_URI', 'bolt://localhost:7778')
driver = GraphDatabase.driver(_uri, auth=(environ.get('BOLT_USER', 'neo4j'), environ.get('BOLT_PASSWORD', 'root123')))


def _generic_get(tx, query, args=None):
    if args:
        return tx.run(query, **args)
    else:
        return tx.run(query)


def _run_filtering(num_particles, source_uris, rank_uris):
    query = """
            MATCH (n) WHERE n.uri IN $uris WITH COLLECT(n) AS nLst
            CALL particlefiltering(nLst, 0, $num_particles) YIELD nodeId, score
            MATCH (n) WHERE n:Movie AND id(n) = nodeId AND n.uri IN $rank RETURN n.uri AS uri, score
    """

    args = {'uris': source_uris, 'num_particles': num_particles, 'rank': rank_uris}

    with driver.session() as session:
        res = session.read_transaction(_generic_get, query, args)

        return {r['uri']: r['score'] for r in res}


class ParticleFilteringRecommender(RecommenderBase):
    def __init__(self, meta: Meta):
        super().__init__(meta)
        self.idx_uri = {idx: uri for uri, idx in meta.uri_idx.items()}

    def fit(self, training: Dict[int, WarmStartUser]):
        pass

    def predict(self, items: List[int], answers: Dict[int, int]) -> Dict[int, float]:
        # Convert items to URIs
        source_uris = [self.idx_uri[idx] for idx, rating in answers.items()]
        rank_uris = [self.idx_uri[idx] for idx in items]

        uri_score = _run_filtering(num_particles=100, source_uris=source_uris, rank_uris=rank_uris)

        return {self.meta.uri_idx[uri]: score for uri, score in uri_score.items()}
