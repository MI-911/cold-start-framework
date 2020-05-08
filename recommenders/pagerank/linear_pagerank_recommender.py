import multiprocessing
from collections import defaultdict
from concurrent.futures._base import wait
from concurrent.futures.process import ProcessPoolExecutor
from concurrent.futures.thread import ThreadPoolExecutor
from copy import deepcopy
from typing import Dict, List

from loguru import logger
from networkx import Graph

from recommenders.base_recommender import RecommenderBase
from recommenders.pagerank.pagerank_recommender import construct_collaborative_graph, \
    construct_knowledge_graph
from recommenders.pagerank.sparse_graph import SparseGraph
from shared.user import WarmStartUser

import numpy as np


class GraphWrapper:
    def __init__(self, training, rating, meta=None, only_kg=False):
        if only_kg:
            self.graph = SparseGraph(construct_knowledge_graph(meta))
        elif not meta:
            self.graph = SparseGraph(construct_collaborative_graph(Graph(), training, rating))
        else:
            self.graph = SparseGraph(construct_collaborative_graph(construct_knowledge_graph(meta), training, rating))

        self.rating_type = rating


def _get_parameters():
    params = {'alphas': np.arange(0.1, 1, 0.15), 'weights': np.arange(-2, 2, 0.25)}

    return params


class LinearPageRankRecommender(RecommenderBase):
    def clear_cache(self):
        pass

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

    def _get_node_weights(self, answers, rating_type):
        rated_entities = []

        for entity_idx, sentiment in answers.items():
            if sentiment == rating_type:
                rated_entities.append(entity_idx)

        unrated_entities = self.entity_indices.difference(rated_entities)

        # Change if needed
        ratings = {1: rated_entities, 0: unrated_entities}
        weights = {1: 1.0, 0: 0. if len(rated_entities) > 0 else 1.}

        # Assign weight to each node depending on their rating
        return {idx: weight for sentiment, weight in weights.items() for idx in ratings[sentiment]}

    def _scores(self, node_weights, items, graph: SparseGraph):
        scores = graph.scores(alpha=self.alpha, personalization=node_weights)
        
        return {item: scores.get(item, 0) for item in items}

    def _optimize_weights(self, predictions, weights, num_graphs):
        best_score = 0
        best_predictions = {}
        
        best_weights = 0
        for weights in self._get_weight_options(weights, num_graphs):
            real_predictions = {}
            for weight, preds in zip(weights, predictions):
                for user, val, ps in preds:
                    if user not in real_predictions:
                        real_predictions[user] = (val, {entity: score * weight for entity, score in ps.items()})
                    else:
                        for k, v in ps.items():
                            real_predictions[user][1][k] += v * weight

            # Use validator to score predictions
            real_predictions = list(real_predictions.values())
            score = self.meta.validator.score(real_predictions, self.meta)

            if score > best_score:
                best_score = score
                best_weights = weights
                best_predictions = deepcopy(real_predictions)

        return best_predictions, best_weights

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
            parameters = _get_parameters()
            for alpha in parameters['alphas']:
                logger.debug(f'Trying with alpha: {alpha}')
                self.alpha = alpha

                preds = self._fit_graphs(training)
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

    def _fit_graphs(self, training: Dict[int, WarmStartUser]):
        return [self._fit(training, graph) for graph in self.graphs]

    def _fit(self, training: Dict[int, WarmStartUser], graph: GraphWrapper):
        predictions = []
        for idx, user in training.items():
            node_weights = self._get_node_weights(user.training, graph.rating_type)
            scores = self._scores(node_weights, user.validation.to_list(), graph.graph)

            predictions.append((idx, user.validation, scores))

        return predictions

    def predict(self, items: List[int], answers: Dict[int, int]) -> Dict[int, float]:
        predictions = defaultdict(int)

        for weight, graph in zip(self.weights, self.graphs):
            node_weights = self._get_node_weights(answers, graph.rating_type)
            scores = self._scores(node_weights, items, graph.graph)

            for item, score in scores.items():
                predictions[item] += score * weight

        return predictions
