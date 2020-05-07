import operator
from functools import reduce
from random import shuffle
from typing import List, Dict

from loguru import logger
import networkx as nx
import numpy as np
from tqdm import tqdm
from recommenders.base_recommender import RecommenderBase
from recommenders.pagerank.sparse_graph import SparseGraph
from shared.meta import Meta
from shared.user import WarmStartUser
from shared.utility import get_combinations, get_top_entities, hashable_lru

RATING_CATEGORIES = {1, 0, -1}


def construct_collaborative_graph(graph: nx.Graph, training: Dict[int, WarmStartUser], rating_type=None):
    for user_id, user in training.items():
        user_id = f'user_{user_id}'
        graph.add_node(user_id, entity=False)

        for entity_idx, sentiment in user.training.items():
            # Skip don't knows if no rating type have been specified
            if rating_type is None and sentiment == 0:
                continue

            if rating_type is None or sentiment == rating_type:
                graph.add_node(entity_idx, entity=True)
                graph.add_edge(user_id, entity_idx, sentiment=sentiment)

    return graph


def construct_knowledge_graph(meta: Meta):
    graph = nx.Graph()

    for triple in meta.triples:
        if triple.head not in meta.uri_idx or triple.tail not in meta.uri_idx:
            logger.warning(f'Could not lookup triple data for {triple.head} or {triple.tail}')

            continue

        head = meta.uri_idx[triple.head]
        tail = meta.uri_idx[triple.tail]

        graph.add_node(head, entity=True)
        graph.add_node(tail, entity=True)
        graph.add_edge(head, tail, type=triple.relation)

    return graph


def get_cache_key(answers):
    return str(sorted(answers.items(), key=lambda x: x[0]))


class PageRankRecommender(RecommenderBase):
    def __init__(self, meta: Meta, ask_limit: int, recommendable_only: bool):
        super().__init__(meta)
        self.predictions_cache = dict()
        self.parameters = None

        self.entity_indices = set()
        self.sparse_graph = None

        # How many of the top-k entities we can ask about in validation
        self.ask_limit = ask_limit

        self.recommendable_only = recommendable_only

    def clear_cache(self):
        self.sparse_graph.scores.cache_clear()

    def construct_graph(self, training: Dict[int, WarmStartUser]):
        raise NotImplementedError()

    def _scores(self, alpha, node_weights, items):
        """
        Produces a ranking of items. If answers is not none, the ranking
        will be reused if produced previously.
        """
        scores = self.sparse_graph.scores(alpha=alpha, personalization=node_weights)

        return {item: scores.get(item, 0) for item in items}

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
        unrated_entities = self.sparse_graph.node_set.difference(rated_entities)

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

        self.sparse_graph = SparseGraph(self.construct_graph(training))

        can_ask_about = get_top_entities(training)
        if self.ask_limit:
            can_ask_about = can_ask_about[:self.ask_limit]

        can_ask_about = set(can_ask_about)

        self.parameters = {
            'alpha': 0.05,
            'importance': {1: 0.95, 0: 0.05, -1: 0.0}
        }

        if not self.parameters:
            parameters = {
                'alpha': np.arange(0.1, 1, 0.15),
                'importance': [
                    {1: 0.95, 0: 0.05, -1: 0.0},
                    {1: 0.80, 0: 0.20, -1: 0.0},
                    {1: 1/3, 0: 1/3, -1: 1/3},
                    {1: 0.45, 0: 0.10, -1: 0.45}
                ]
            }

            combinations = get_combinations(parameters)

            results = list()

            for combination in combinations:
                logger.debug(f'Trying {combination}')

                self.parameters = combination

                predictions = list()
                for _, user in tqdm(training.items()):
                    user_answers = {idx: rating for idx, rating in user.training.items() if idx in can_ask_about}
                    prediction = self.predict(user.validation.to_list(), user_answers)

                    predictions.append((user.validation, prediction))

                score = self.meta.validator.score(predictions, self.meta)
                results.append((combination, score))

                logger.info(f'Score: {score}')

                self.clear_cache()

            self.parameters = sorted(results, key=operator.itemgetter(1), reverse=True)[0][0]

            logger.info(f'Found optimal: {self.parameters}')

    def predict(self, items: List[int], answers: Dict[int, int]) -> Dict[int, float]:
        if self.recommendable_only:
            answers = {idx: sentiment for idx, sentiment in answers.items() if idx in self.meta.recommendable_entities}

        return self._scores(self.parameters['alpha'],
                            self.get_node_weights(answers, self.parameters['importance']), items)
