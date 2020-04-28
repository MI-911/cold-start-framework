import multiprocessing
from collections import defaultdict
from concurrent.futures._base import wait
from concurrent.futures.process import ProcessPoolExecutor
from concurrent.futures.thread import ThreadPoolExecutor
from copy import deepcopy
from functools import reduce
from typing import Dict, List

from loguru import logger

from recommenders.base_recommender import RecommenderBase
from recommenders.pagerank.pagerank_recommender import PageRankRecommender, construct_collaborative_graph, \
    RATING_CATEGORIES, construct_knowledge_graph
from shared.meta import Meta
from shared.user import WarmStartUser
from networkx import Graph, pagerank_scipy


class GraphWrapper:
    def __init__(self, training, rating_type, meta=None, only_kg=False):
        if only_kg:
          self.graph = construct_knowledge_graph(meta)
        elif meta is None:
            self.graph = construct_collaborative_graph(Graph(), training, rating_type)
        else:
            self.graph = construct_collaborative_graph(construct_knowledge_graph(meta), training, rating_type)
        self.rating_type = rating_type


class LinearPageRankRecommender(RecommenderBase):
    def __init__(self, meta):
        RecommenderBase.__init__(self, meta)

        # Entities
        self.entity_indices = set()
        self.graphs = None

        # Parameters
        self.alpha = 0
        self.weights = None

        self.optimal_params = None

    def construct_graph(self, training: Dict[int, WarmStartUser]) -> List[GraphWrapper]:
        raise NotImplementedError()

    def _get_parameters(self):
        params = {'alphas': [0.2, 0.50, 0.85],
                  'weights': [-2, -1, -0.5, -0.4, -0.3, -0.2, 0, 0.2, 0.3, 0.4, 0.5, 1, 2]}

        return params

    def _get_node_weights(self, answers, rating_type):
        rated_entities = []

        for entity_idx, sentiment in answers.items():
            if sentiment == rating_type:
                rated_entities.append(entity_idx)

        unrated_entities = self.entity_indices.difference(rated_entities)

        # Change if needed
        ratings = {1: rated_entities, 0: unrated_entities}
        weights = {1: 1.0,
                   0: 0. if len(rated_entities) > 0 else 1.}

        # Assign weight to each node depending on their rating
        return {idx: weight for sentiment, weight in weights.items() for idx in ratings[sentiment]}

    def _scores(self, node_weights, items, graph: Graph):
        scores = pagerank_scipy(graph, alpha=self.alpha, personalization=node_weights).items()
        scores = {item: score for item, score in scores}
        return {item: scores.get(item, 0) for item in items}

    def _optimize_weights(self, predictions, weights, num_graphs):
        best_score = 0
        best_preds = {}
        best_weights = 0
        for weights in self._get_weight_options(weights, num_graphs):
            real_predictions = {}
            for weight, preds in zip(weights, predictions):
                for user, val, ps in preds:
                    if user not in real_predictions:
                        real_predictions[user] = (val, {k: v * weight for k, v in ps.items()})
                    else:
                        for k, v in ps.items():
                            real_predictions[user][1][k] += v * weight
            preds = list(real_predictions.values())
            score = self.meta.validator.score(preds, self.meta)

            if score > best_score:
                best_score = score
                best_weights = weights
                best_preds = deepcopy(preds)

        return best_preds, best_weights

    def _get_weight_options(self, weights, num_graphs):
        if num_graphs <= 1:
            return weights
        else:
            options = []
            for weight in weights:
                o = self._get_weight_options(weights, num_graphs - 1)
                for option in o:
                    if isinstance(option, tuple):
                        options.append((weight, *option))
                    else:
                        options.append((weight, option))

            return options

    def _set_parameters(self, parameters):
        self.alpha = parameters['alpha']
        self.weights = parameters['weights']

    def fit(self, training: Dict[int, WarmStartUser]):
        # Get sentiments and entities
        sentiments = []
        for _, user in training.items():
            for entity, sentiment in user.training.items():
                self.entity_indices.add(entity)
                sentiments.append(sentiment)

        self.graphs = self.construct_graph(training)

        if self.optimal_params is None:
            best_score = -1
            best_params = None
            parameters = self._get_parameters()
            for alpha in parameters['alphas']:
                logger.debug(f'Trying with alpha: {alpha}')
                self.alpha = alpha

                preds = self._multi_fit(training)
                preds, weights = self._optimize_weights(preds, parameters['weights'], len(self.graphs))
                logger.debug(f'Best weights with rating for alpha {alpha}: '
                             f'{[(weight, graph.rating_type) for weight, graph in zip(weights, self.graphs)]}')

                score = self.meta.validator.score(preds, self.meta)

                if score > best_score:
                    logger.debug(f'New best with score: {score} and params, alpha: {alpha}, weights:{weights}')
                    best_score = score
                    best_params = {'alpha': alpha, 'weights': weights}

            self.optimal_params = best_params

        self._set_parameters(self.optimal_params)

    def _multi_fit(self, training: Dict[int, WarmStartUser]):
        futures = []
        outer_workers = len(self.graphs)
        inner_workers = multiprocessing.cpu_count() // outer_workers
        inner_workers = inner_workers if inner_workers != 0 else 1

        with ThreadPoolExecutor(max_workers=outer_workers) as executor:
            for graph in self.graphs:
                futures.append(executor.submit(self._inner_multi_fit, training, graph, inner_workers))

            wait(futures)

        return [f.result() for f in futures]

    def _inner_multi_fit(self, training: Dict[int, WarmStartUser], graph: GraphWrapper, workers: int):
        lst = list(training.items())
        chunks = [lst[i::workers] for i in range(workers)]
        futures = []
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for chunk in chunks:
                futures.append(executor.submit(self._fit, dict(chunk), graph))

        return [r for future in futures for r in future.result()]

    def _fit(self, training: Dict[int, WarmStartUser], graph: GraphWrapper):
        predictions = []
        for user, warm in training.items():
            node_weights = self._get_node_weights(warm.training, graph.rating_type)
            prediction = self._scores(node_weights, warm.validation.to_list(), graph.graph)
            predictions.append((user, warm.validation, prediction))

        return predictions

    def predict(self, items: List[int], answers: Dict[int, int]) -> Dict[int, float]:
        predictions = defaultdict(int)
        for weight, graph in zip(self.weights, self.graphs):
            node_weights = self._get_node_weights(answers, graph.rating_type)
            preds = self._scores(node_weights, items, graph.graph)

            for item, score in preds.items():
                predictions[item] += score * weight

        return predictions