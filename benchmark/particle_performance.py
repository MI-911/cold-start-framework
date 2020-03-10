import json
import pickle
import time
from os import environ
from random import shuffle

import numpy as np
import tqdm
from neo4j import GraphDatabase
from loguru import logger

num_particles = [10, 25, 50, 75, 100, 200, 500, 1000]

_uri = environ.get('BOLT_URI', 'bolt://localhost:7778')
driver = GraphDatabase.driver(_uri, auth=("neo4j", "root123"))


def _generic_get(tx, query, args=None):
    if args:
        return tx.run(query, **args)
    else:
        return tx.run(query)


def run_ppr(damping_factor, cutoff, source_uris, rank):
    query = """
            MATCH (n:Movie) WHERE n.uri IN $uris WITH collect(n) AS movies
            CALL algo.pageRank.stream(
                null,
                null,
              {iterations: 50, dampingFactor: $damping, sourceNodes: movies, direction: 'BOTH'}
            )
            YIELD nodeId, score
            MATCH (n) WHERE n:Movie AND id(n) = nodeId AND n.uri IN $rank RETURN n.uri AS uri, score
            ORDER BY score DESC
            LIMIT $cutoff
        """

    args = {'uris': source_uris, 'damping': damping_factor, 'cutoff': cutoff, 'rank': rank}

    with driver.session() as session:
        res = session.read_transaction(_generic_get, query, args)
        res = [r['uri'] for r in res]

    return res


def run_filtering(num_particles, cutoff, source_uris, rank):
    query = """
            MATCH (n) WHERE n.uri IN $uris WITH COLLECT(n) AS nLst
            CALL particlefiltering(nLst, 0, $num_particles) YIELD nodeId, score
            MATCH (n) WHERE n:Movie AND id(n) = nodeId AND n.uri IN $rank RETURN n.uri AS uri, score
            ORDER BY score DESC
            LIMIT $cutoff
    """

    args = {'uris': source_uris, 'num_particles': num_particles, 'cutoff': cutoff, 'rank': rank}

    with driver.session() as session:
        res = session.read_transaction(_generic_get, query, args)
        res = [r['uri'] for r in res]

    return res


def benchmark_particles():
    training = pickle.load(open('../data/basic/split_0/training.pkl', 'rb'))
    meta = pickle.load(open('../data/basic/split_0/meta.pkl', 'rb'))

    idx_uri = {v: k for k, v in meta.uri_idx.items()}
    cutoff = 10
    results = dict()

    for particles in num_particles:
        time_taken = list()
        hits = list()

        for user, data in tqdm.tqdm(training.items()):
            start_time = time.time()
            positive_uri = idx_uri[data.validation['positive']]
            liked_uris = [idx_uri[idx] for idx, rating in data.training.items() if rating == 1]
            uris_to_rank = [positive_uri] + [idx_uri[idx] for idx in data.validation['negative']]
            shuffle(uris_to_rank)

            assert not set(liked_uris).intersection(set(uris_to_rank))

            if not liked_uris or not uris_to_rank:
                continue

            # ranked_list = run_ppr(0.25, cutoff, liked_uris, uris_to_rank)
            ranked_list = run_filtering(particles, cutoff, liked_uris, uris_to_rank)

            hits.append(positive_uri in ranked_list)
            time_taken.append(time.time() - start_time)

        logger.info(f'Finished n_particles={particles}')

        results[particles] = {
            'hr': np.mean(hits),
            'time': np.mean(time_taken)
        }

        logger.info(results[particles])

        json.dump(results, open('particles_benchmark.json', 'w'))


if __name__ == '__main__':
    benchmark_particles()
