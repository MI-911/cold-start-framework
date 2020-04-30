import operator
from functools import reduce
from random import shuffle
from typing import List, Dict

from loguru import logger
from networkx import Graph, pagerank_scipy
from tqdm import tqdm

from recommenders.base_recommender import RecommenderBase
from shared.meta import Meta
from shared.user import WarmStartUser
from shared.utility import get_combinations

RATING_CATEGORIES = {1, 0, -1}


def construct_collaborative_graph(graph: Graph, training: Dict[int, WarmStartUser], rating_type=None):
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
    graph = Graph()

    for triple in meta.triples:
        head = meta.uri_idx[triple.head]
        tail = meta.uri_idx[triple.tail]

        graph.add_node(head, entity=True)
        graph.add_node(tail, entity=True)
        graph.add_edge(head, tail, type=triple.relation)

    return graph


def get_cache_id(answers):
    return str(sorted(answers.items(), key=lambda x: x[0]))


class PageRankRecommender(RecommenderBase):
    def __init__(self, meta: Meta):
        super().__init__(meta)
        self.graph = None
        self.entity_indices = set()
        self.optimal_params = None
        self.predictions_cache = {}

    def construct_graph(self, training: Dict[int, WarmStartUser]):
        raise NotImplementedError()

    def _scores(self, alpha, node_weights, items, answers=None):
        """
        Produces a ranking of items. If answers is not none, the ranking
        will be reused if produced previously.
        """
        if not answers:
            scores = pagerank_scipy(self.graph, alpha=alpha, personalization=node_weights).items()
            return {item: score for item, score in scores if item in items}

        cache_id = get_cache_id(answers)
        if cache_id not in self.predictions_cache:
            scores = pagerank_scipy(self.graph, alpha=alpha, personalization=node_weights).items()
            self.predictions_cache[cache_id] = {entity: score for entity, score in scores}

        return {item: self.predictions_cache[cache_id].get(item, 0.0) for item in items}

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

        if not self.optimal_params:
            parameters = {
                'alpha': [0.05, 0.15, 0.5, 0.85],
                'importance': [
                    {1: 0.95, 0: 0.05, -1: 0.0},
                    {1: 1/3, 0: 1/3, -1: 1/3},
                    {1: 0.45, 0: 0.10, -1: 0.45}
                ]
            }

            combinations = get_combinations(parameters)

            results = list()

            validation_users = list(training.items())
            shuffle(validation_users)

            for combination in combinations:
                logger.debug(f'Trying {combination}')

                predictions = list()
                for _, user in tqdm(validation_users[:int(len(validation_users) * 0.25)]):
                    node_weights = self.get_node_weights(user.training, combination['importance'])
                    prediction = self._scores(combination['alpha'], node_weights, user.validation.to_list())

                    predictions.append((user.validation, prediction))

                score = self.meta.validator.score(predictions, self.meta)
                results.append((combination, score))

                logger.info(f'Score: {score}')

            self.optimal_params = sorted(results, key=operator.itemgetter(1), reverse=True)[0][0]
            logger.info(f'Found optimal: {self.optimal_params}')

    def predict(self, items: List[int], answers: Dict[int, int]) -> Dict[int, float]:
        return self._scores(self.optimal_params['alpha'],
                            self.get_node_weights(answers, self.optimal_params['importance']), items, answers=answers)
