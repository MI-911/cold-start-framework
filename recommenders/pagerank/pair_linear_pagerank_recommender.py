from collections import defaultdict
from concurrent.futures._base import wait
from concurrent.futures.process import ProcessPoolExecutor
from concurrent.futures.thread import ThreadPoolExecutor
from copy import deepcopy
from random import shuffle
from typing import Dict, List, Tuple

import numpy as np
from loguru import logger
from networkx import Graph
from torch import nn
import torch as tt
from tqdm import tqdm

from recommenders.base_recommender import RecommenderBase
from recommenders.pagerank.pagerank_recommender import construct_collaborative_graph, \
    construct_knowledge_graph
from recommenders.pagerank.sparse_graph import SparseGraph
from shared.enums import Sentiment
from shared.ranking import Ranking
from shared.user import WarmStartUser
from shared.utility import hashable_lru


class GraphWrapper:
    def __init__(self, training, rating, meta, ask_limit, only_kg=False, use_meta=False):
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

    def get_score(self, answers, items):
        scores = self._all_scores(answers)

        return {item: scores.get(item, 0) for item in items}

    @hashable_lru(maxsize=1024)
    def _all_scores(self, answers):
        node_weights = self._get_node_weights(answers)

        return self.graph.scores(alpha=self.alpha, personalization=node_weights)

    def clear_cache(self):
        self._all_scores.cache_clear()

    def _get_node_weights(self, answers):
        answers = {k: v for k, v in answers.items() if v == self.rating_type and k in self.can_ask_about}
        return self._get_node_weights_cached(answers)

    def _get_node_weights_cached(self, answers):
        rated_entities = list(answers.keys())

        unrated_entities = self.entity_indices.difference(rated_entities)

        # Change if needed
        ratings = {1: rated_entities, 0: unrated_entities}
        weights = {1: 1.0, 0: 0. if len(rated_entities) > 0 else 1.}

        # Assign weight to each node depending on their rating
        return {idx: weight for sentiment, weight in weights.items() for idx in ratings[sentiment]}


def _get_parameters():
    params = {'alphas': np.arange(0.1, 1, 0.15), 'weights': np.arange(-2, 2, 0.5)}

    return params


class PairwiseLinear(nn.Module):
    def __init__(self, num_graphs):
        super().__init__()
        self.weights = nn.Linear(num_graphs, num_graphs, bias=False)

    def forward(self, scores):
        x = self.weights(scores)
        x = tt.sum(x, dim=1)
        return x


class PairLinearPageRankRecommender(RecommenderBase):
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
        self.model = None
        self.optimizer = None
        self.loss_func = nn.MarginRankingLoss(margin=1.0)
        self.batch_size = 64

    def construct_graph(self, training: Dict[int, WarmStartUser]) -> List[GraphWrapper]:
        raise NotImplementedError()

    def _optimize_weights(self, batches, predictions, epochs=100) -> List[Tuple[Ranking, Dict[int, float]]]:
        target = tt.ones(self.batch_size, dtype=tt.float)
        t = tqdm(range(epochs), total=epochs)
        for epoch in t:
            running_loss = tt.tensor(0.)
            count = tt.tensor(0.)
            shuffle(batches)

            for pos, neg in batches:
                self.optimizer.zero_grad()
                pos_val = self.model(pos)
                neg_val = self.model(neg)

                loss = self.loss_func(pos_val, neg_val, target)
                loss.backward()
                self.optimizer.step()

                with tt.no_grad():
                    running_loss += loss
                    count += tt.tensor(1.)
                    t.set_description(f'Loss: {running_loss / count:.4f}')

        preds = []
        with tt.no_grad():
            for idx, val, scores in predictions:
                p = {entity: self.model(tt.tensor(self.get_score(scores, entity)).unsqueeze(0))
                     for entity in val.to_list()}
                preds.append((val, p))

        return preds

    def _set_parameters(self, parameters):
        self.alpha = parameters['alpha']
        self.model = PairwiseLinear(len(self.graphs))
        self.optimizer = tt.optim.Adam(self.model.parameters())

        for graph in self.graphs:
            graph.alpha = self.alpha

    @staticmethod
    def get_score(scores, entity):
        return [score[entity] for score in scores]

    def _get_batches(self, preds):
        data = []
        for idx, val, scores in preds:
            pos_samples = val.sentiment_samples[Sentiment.POSITIVE]
            neg_samples = [sample for sample in val.to_list() if sample not in pos_samples]
            for pos_sample in pos_samples:
                for neg_sample in neg_samples:

                    data.append([self.get_score(scores, pos_sample), self.get_score(scores, neg_sample)])

        shuffle(data)
        batches = []
        for batch_n in range(len(data) // self.batch_size):
            batch = data[self.batch_size * batch_n:self.batch_size * (batch_n + 1)]
            batches.append([tt.tensor(a) for a in zip(*batch)])

        return batches

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
                parameter = {'alpha': alpha}
                self._set_parameters(parameter)

                preds = self._fit(training)
                batches = self._get_batches(preds)
                score = self.meta.validator.score(self._optimize_weights(batches, preds, 50), self.meta)

                # Clear cache
                self.clear_cache()

                if score > best_score:
                    logger.debug(f'New best with score: {score} and params, alpha: {alpha}')
                    best_score = score
                    best_params = parameter

            self.optimal_params = best_params

        self._set_parameters(self.optimal_params)
        preds = self._fit(training)
        batches = self._get_batches(preds)
        self._optimize_weights(batches, preds, 100)

    def _fit(self, training: Dict[int, WarmStartUser]):
        predictions = []
        for idx, user in tqdm(training.items(), total=len(training)):
            scores = [graph.get_score(user.training, user.validation.to_list()) for graph in self.graphs]

            predictions.append((idx, user.validation, scores))

        return predictions

    def predict(self, items: List[int], answers: Dict[int, int]) -> Dict[int, float]:
        with tt.no_grad():
            scores = [graph.get_score(answers, items) for graph in self.graphs]
            predictions = {entity: self.model(tt.tensor(self.get_score(scores, entity)).unsqueeze(0))
                           for entity in items}

        return predictions
