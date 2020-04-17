from concurrent.futures._base import wait
from concurrent.futures.process import ProcessPoolExecutor
from functools import reduce
from typing import Dict, List

from loguru import logger

from recommenders.base_recommender import RecommenderBase
from recommenders.pagerank.pagerank_recommender import PageRankRecommender, construct_collaborative_graph, \
    RATING_CATEGORIES
from shared.user import WarmStartUser
from networkx import Graph, pagerank_scipy


class LinearPageRankRecommender(RecommenderBase):
    def __init__(self, meta):
        RecommenderBase.__init__(self, meta)

        # Graphs
        self.graph_like = None
        self.graph_dislike = None
        self.graph_dont_know = None

        # Entities
        self.entity_indices = set()

        # Parameters
        self.alpha = 0
        self.lw = 0
        self.dlw = 0
        self.dkw = 0

        self.optimal_params = None

    @staticmethod
    def _get_parameters():
        params = []
        for alpha in [0.25, 0.50, 0.85]:
            for like_weight in [0, 0.5, 1, 2]:
                for dislike_weight in [0, -0.5, -1, -2]:
                    for dont_know_weight in [-2, -1, -0.5, 0, 0.5, 1, 2]:
                        params.append({'a': alpha, 'lw': like_weight, 'dlw': dislike_weight,
                                       'dkw': dont_know_weight})
        return params

    def _set_params(self, params):
        self.alpha = params['a']
        self.lw = params['lw']
        self.dlw = params['dlw']
        self.dkw = params['dkw']

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

    def fit(self, training: Dict[int, WarmStartUser]):
        for _, user in training.items():
            for entity in user.training.keys():
                self.entity_indices.add(entity)

        self.graph_like = construct_collaborative_graph(Graph(), training, 1)
        self.graph_dislike = construct_collaborative_graph(Graph(), training, -1)
        self.graph_dont_know = construct_collaborative_graph(Graph(), training, 0)

        if self.optimal_params is None:
            best_score = -1
            best_params = None
            for params in self._get_parameters():
                logger.debug(f'Trying with params: {params}')
                self._set_params(params)
                preds = self._multi_fit(training)
                score = self.meta.validator.score(preds, self.meta)

                if score > best_score:
                    logger.debug(f'Parameters were better with score: {score}')
                    best_score = score
                    best_params = params

            self.optimal_params = best_params

        self._set_params(self.optimal_params)

    def _multi_fit(self, training: Dict[int, WarmStartUser]):
        futures = []
        # with ProcessPoolExecutor(max_workers=3) as executor:
        #     futures.append(executor.submit(self._fit, training, self.graph_like, 1))
        #     futures.append(executor.submit(self._fit, training, self.graph_dislike, -1))
        #     futures.append(executor.submit(self._fit, training, self.graph_dont_know, 0))
        #
        #     wait(futures)
        tmp = self._fit(training, self.graph_dont_know, 0)
        preds = []
        for future in futures:
            res = future.result()
            preds.append(res)

        preds = list(zip(*preds))

        predictions = []
        for (user, likes), (_, dislikes), (_, dontknows) in preds:
            p = {}
            for k in likes.keys():
                p[k] = likes[k] * self.lw + dislikes[k] * self.dlw + dontknows[k] * self.dkw

            predictions.append((user, p))

        return predictions

    def _fit(self, training: Dict[int, WarmStartUser], graph: Graph, ratings_type):
        predictions = []
        for _, warm in training.items():
            node_weights, any_ratings = self.get_node_weights(warm.training, ratings_type)
            prediction = self._scores(node_weights, warm.validation.to_list(), graph)
            predictions.append((warm.validation, prediction))

        return predictions

    def predict(self, items: List[int], answers: Dict[int, int]) -> Dict[int, float]:
        pass
