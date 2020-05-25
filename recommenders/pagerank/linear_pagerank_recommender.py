from collections import defaultdict
from concurrent.futures._base import wait
from concurrent.futures.process import ProcessPoolExecutor
from concurrent.futures.thread import ThreadPoolExecutor
from copy import deepcopy
from typing import Dict, List

import numpy as np
from loguru import logger
from networkx import Graph
from tqdm import tqdm

from recommenders.base_recommender import RecommenderBase
from recommenders.pagerank.pagerank_recommender import construct_collaborative_graph, \
    construct_knowledge_graph
from recommenders.pagerank.sparse_graph import SparseGraph
from shared.user import WarmStartUser
from shared.utility import hashable_lru


class GraphWrapper:
    def __init__(self, training, rating, meta, ask_limit, only_kg=False, use_meta=False, normalize=False):
        if only_kg:
            self.graph = SparseGraph(construct_knowledge_graph(meta))
        elif not use_meta:
            self.graph = SparseGraph(construct_collaborative_graph(Graph(), training, rating))
        else:
            self.graph = SparseGraph(construct_collaborative_graph(construct_knowledge_graph(meta), training, rating))

        self.rating_type = rating
        self.meta = meta
        self.entity_indices = {idx for _, warm in training.items() for idx, _ in warm.training.items()}
        self.can_ask_about = set(self.meta.get_question_candidates(training, limit=ask_limit))
        self.alpha = None
        self.normalize = normalize

    def get_score(self, answers, items):
        scores = self._all_scores(answers)
        scores = {item: scores.get(item, 0) for item in items}
        if self.normalize:
            return self._normalize_scores(scores)
        else:
            return scores

    @hashable_lru(maxsize=1024)
    def _all_scores(self, answers):
        node_weights = self._get_node_weights(answers)
        return self.graph.scores(alpha=self.alpha, personalization=node_weights)

    def clear_cache(self):
        self._all_scores.cache_clear()
        self._get_node_weights_cached.cache_clear()

    def _get_node_weights(self, answers):
        answers = {k: v for k, v in answers.items() if v == self.rating_type and k in self.can_ask_about}
        return self._get_node_weights_cached(answers)

    @hashable_lru()
    def _get_node_weights_cached(self, answers):
        rated_entities = list(answers.keys())

        unrated_entities = self.entity_indices.difference(rated_entities)

        # Change if needed
        ratings = {1: rated_entities, 0: unrated_entities}
        weights = {1: 1.0, 0: 0. if len(rated_entities) > 0 else 1.}

        # Assign weight to each node depending on their rating
        return {idx: weight for sentiment, weight in weights.items() for idx in ratings[sentiment]}

    def _normalize_scores(self, scores):
        factor = max(scores.values())
        return {k: v/factor for k, v in scores.items()}


def _get_parameters():
    params = {'alphas': np.arange(0.1, 1, 0.15), 'weights': np.arange(-2, 2, 0.5)}

    return params


class LinearPageRankRecommender(RecommenderBase):
    def clear_cache(self):
        for graph in self.graphs:
            graph.clear_cache()

    def __init__(self, meta, ask_limit: int):
        RecommenderBase.__init__(self, meta)

        # Entities
        self.entity_indices = set()
        self.graphs = None

        # Parameters
        self.alpha = 0
        self.weights = None

        self.optimal_params = None

        self.ask_limit = ask_limit
        self.can_ask_about = None

    def construct_graph(self, training: Dict[int, WarmStartUser]) -> List[GraphWrapper]:
        raise NotImplementedError()

    def _optimize_weights(self, predictions, weights, num_graphs):
        weight_ops = self._get_weight_options(weights, num_graphs)
        num_preds = len(predictions)
        num_users = len(predictions[0])
        num_items = len(predictions[0][0][2])

        data_score = np.zeros((num_preds, num_users, num_items))
        user_map = {}
        user_val_map = {}
        user_entities_map = np.empty((num_users, num_items), dtype=np.int)
        user_idx = 0
        for i, prediction in enumerate(predictions):
            for user, val, preds in prediction:
                if user not in user_map:
                    user_map[user] = user_idx
                    user_val_map[user_idx] = val
                    user_idx += 1

                user = user_map[user]
                preds = sorted(preds.items(), key=lambda x: x[0])
                for j, (_, score) in enumerate(preds):
                    data_score[i][user][j] = score

                if user not in user_entities_map:
                    user_entities_map[user] = [p[0] for p in preds]

        validations = np.array([val for _, val in sorted(user_val_map.items(), key=lambda x: x[0])])

        workers = 1
        chunks = [weight_ops[w::workers] for w in range(workers)]

        futures = []
        with ProcessPoolExecutor(max_workers=workers) as e:
            for chunk in chunks:
                futures.append(e.submit(self._weight_score, num_users, num_items, chunk,
                                        data_score, validations, user_entities_map))

        wait(futures)

        scores = np.array([future.result() for future in futures])
        x, y = 0, 0
        best = 0
        for i, score in enumerate(scores):
            arg = np.argmax(score)
            current = score[arg]
            if current > best:
                x = i
                y = arg

        return scores[x][y], chunks[x][y]

    def _weight_score(self, num_users, num_items, weight_ops, data_score, validations, user_entities_map):
        scores = np.zeros((len(weight_ops)))
        for i, weights in tqdm(enumerate(weight_ops), total=len(weight_ops)):
            linear_preds = np.zeros((num_users, num_items), dtype=np.float)
            for preds_index, weight in enumerate(weights):
                linear_preds += data_score[preds_index] * weight

            linear_preds = list(zip(validations, [{**dict(zip(entities, preds))}
                                                  for entities, preds in zip(user_entities_map, linear_preds)]))

            scores[i] = self.meta.validator.score(linear_preds, self.meta)

        return scores

    def _get_weight_options(self, weights, num_graphs):
        if num_graphs <= 1:
            return weights
        else:
            options = []
            for weight in weights:
                for option in self._get_weight_options(weights, num_graphs - 1):
                    if isinstance(option, tuple):
                        options.append((weight, *option))
                    else:
                        options.append((weight, option))

            return options

    def _set_parameters(self, parameters):
        self.alpha = parameters['alpha']
        self.weights = parameters['weights']

        for graph in self.graphs:
            graph.alpha = self.alpha

    def fit(self, training: Dict[int, WarmStartUser]):
        self.can_ask_about = set(self.meta.get_question_candidates(training, limit=self.ask_limit))

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

                # Update alpha
                self.alpha = alpha
                for graph in self.graphs:
                    graph.alpha = alpha

                preds = self._fit_graphs(training)
                score, weights = self._optimize_weights(preds, parameters['weights'], len(self.graphs))
                logger.debug(f'Best weights with rating for alpha {alpha}: '
                             f'{[(weight, graph.rating_type) for weight, graph in zip(weights, self.graphs)]}')

                # Clear cache
                self.clear_cache()

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
        for idx, user in tqdm(training.items(), total=len(training)):
            scores = graph.get_score(user.training, user.validation.to_list())

            predictions.append((idx, user.validation, scores))

        return predictions

    def predict(self, items: List[int], answers: Dict[int, int]) -> Dict[int, float]:
        predictions = defaultdict(int)

        for weight, graph in zip(self.weights, self.graphs):
            scores = graph.get_score(answers, items)

            for item, score in scores.items():
                predictions[item] += score * weight

        return predictions
