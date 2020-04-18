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
    RATING_CATEGORIES
from shared.user import WarmStartUser
from networkx import Graph, pagerank_scipy


class GraphWrapper:
    def __init__(self, training, rating_type):
        self.graph =  construct_collaborative_graph(Graph(), training, rating_type)
        self.rating_type = rating_type


class LinearPageRankRecommender(RecommenderBase):
    def __init__(self, meta):
        RecommenderBase.__init__(self, meta)

        # Entities
        self.entity_indices = set()
        self.graphs = None

        # Parameters
        self.alpha = 0
        self.graph_weights = None

        self.optimal_params = None

    def _get_parameters(self):
        params = {'alphas': [0.2, 0.50, 0.85],
                  'weights': [-2, -1, -0.5, -0.4, -0.3, -0.2, 0, 0.2, 0.3, 0.4, 0.5, 1, 2]}

        return params

    def get_node_weights(self, answers, rating_type):
        rated_entities = []

        for entity_idx, sentiment in answers.items():
            if sentiment == rating_type:
                rated_entities.append(entity_idx)

        unrated_entities = self.entity_indices.difference(rated_entities)

        # Change if needed
        ratings = {1: rated_entities, 0: unrated_entities}
        weights = {1: 1 / len(rated_entities) if len(rated_entities) > 0 else 0,
                   0: 0}

        # Assign weight to each node depending on their rating
        return {idx: weight for sentiment, weight in weights.items() for idx in ratings[sentiment]}, \
               len(rated_entities) > 0

    def _scores(self, node_weights, items, graph: Graph):
        scores = pagerank_scipy(graph, alpha=self.alpha, personalization=node_weights).items()

        return {item: score for item, score in scores if item in items}

    def _optimize_weights(self, predictions, weights, num_graphs):
        best_score = 0
        best_preds = {}
        best_weights = 0
        for weights in self._get_weight_options(weights, num_graphs):
            real_predictions = {}
            for weight, (user, val, ps) in zip(weights, predictions[num_graphs-1]):
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
        self.graph_weights = parameters['weights']

    def fit(self, training: Dict[int, WarmStartUser]):
        # Get sentiments and entities
        sentiments = []
        for _, user in training.items():
            for entity, sentiment in user.training.items():
                self.entity_indices.add(entity)
                sentiments.append(sentiment)

        # Create graphs
        sentiments = set(sentiments)
        graphs = []
        for sentiment in sentiments:
            graphs.append(GraphWrapper(training, sentiment))

        self.graphs = graphs

        if self.optimal_params is None:
            best_score = -1
            best_params = None
            parameters = self._get_parameters()
            for alpha in parameters['alphas']:
                logger.debug(f'Trying with alpha: {alpha}')
                self.alpha = alpha

                preds = self._multi_fit(training, graphs)
                preds, weights = self._optimize_weights(preds, parameters['weights'], len(graphs))
                logger.debug(f'Best weights for alpha {alpha}: {weights}')

                score = self.meta.validator.score(preds, self.meta)

                if score > best_score:
                    logger.debug(f'Parameters were better with score: {score}')
                    best_score = score
                    best_params = {'alpha': alpha, 'weights': weights}

            self.optimal_params = best_params

        self._set_parameters(self.optimal_params)

    def _multi_fit(self, training: Dict[int, WarmStartUser], graphs: List[GraphWrapper]):
        futures = []
        outer_workers = len(graphs)
        inner_workers = multiprocessing.cpu_count() // outer_workers
        inner_workers = inner_workers if inner_workers != 0 else 1

        with ThreadPoolExecutor(max_workers=outer_workers) as executor:
            for graph in graphs[::-1]:
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
            node_weights, any_ratings = self.get_node_weights(warm.training, graph.rating_type)
            if any_ratings:
                prediction = self._scores(node_weights, warm.validation.to_list(), graph.graph)
            else:
                prediction = {k: 0 for k in warm.validation.to_list()}
            predictions.append((user, warm.validation, prediction))

        return predictions

    def predict(self, items: List[int], answers: Dict[int, int]) -> Dict[int, float]:
        predictions = defaultdict(int)
        for weight, graph in zip(self.graph_weights, self.graphs):
            node_weights, any_ratings = self.get_node_weights(answers, graph.rating_type)
            if any_ratings:
                prediction = self._scores(node_weights, items, graph.graph)
            else:
                prediction = {k: 0 for k in items}

            for k, v in prediction.items():
                predictions[k] += v * weight

        return predictions
