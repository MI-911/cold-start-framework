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
from recommenders.pagerank.linear_pagerank_recommender import GraphWrapper
from recommenders.pagerank.pagerank_recommender import construct_collaborative_graph, \
    construct_knowledge_graph
from recommenders.pagerank.sparse_graph import SparseGraph
from shared.enums import Sentiment
from shared.ranking import Ranking
from shared.user import WarmStartUser
from shared.utility import hashable_lru

def _get_parameters():
    params = []

    for alpha in [0.7]:#np.arange(0.1, 1, 0.15):
        params.append({'alpha': alpha})

    return params


class PairwiseLinear(nn.Module):
    def __init__(self, num_graphs):
        super().__init__()
        self.weights = nn.Linear(num_graphs, 1, bias=True)
        # self.weights = nn.Parameter(tt.rand(1, num_graphs), requires_grad=True)

    def forward(self, scores):
        # x = tt.mul(self.weights, scores)
        x = self.weights(scores)
        # x = tt.sum(x, dim=1)
        return x
        # return tt.sigmoid(x)


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
        self.positive_optimizer = None
        self.negative_optimizer = None
        self.positive_loss_func = nn.MSELoss()
        self.negative_loss_func = nn.MarginRankingLoss(margin=.001)
        self.batch_size = 64

    def construct_graph(self, training: Dict[int, WarmStartUser]) -> List[GraphWrapper]:
        raise NotImplementedError()

    def _optimize_weights(self, batches, predictions, epochs=100) -> List[Tuple[Ranking, Dict[int, float]]]:
        for epoch in range(epochs):
            running_loss = tt.tensor(0.)
            count = tt.tensor(0.)
            shuffle(batches)

            t = tqdm(batches)
            for pos, neg, target in t:
                self.positive_optimizer.zero_grad()
                self.negative_optimizer.zero_grad()
                pos_val = self.model(pos)
                neg_val = self.model(neg)

                if tt.any(target.bool()):
                    loss = self.negative_loss_func(pos_val, neg_val, target)
                    loss.backward()
                    self.negative_optimizer.step()
                else:
                    loss = self.positive_loss_func(pos_val, neg_val)
                    loss.backward()
                    self.positive_optimizer.step()

                with tt.no_grad():
                    running_loss += loss
                    count += tt.tensor(1.)
                    t.set_description(f'[Epoch {epoch}] Loss: {running_loss / count:.16f}')

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
        self.positive_optimizer = tt.optim.Adam(self.model.parameters())
        self.negative_optimizer = tt.optim.Adam(self.model.parameters())

        for graph in self.graphs:
            graph.alpha = self.alpha

    @staticmethod
    def get_score(scores, entity):
        return [score[entity] for score in scores]

    def _get_batches(self, preds, training):
        positive_data = []
        negative_data = []
        for idx, val, scores in preds:
            # user = training[idx].validation.sentiment_samples[Sentiment.UNSEEN]
            # extra = {entity: 0 for entity in user}
            for sample_one, rating_one in val.items():
                for sample_two, rating_two in val.items():
                    if sample_one == sample_two:
                        continue

                    if rating_one == rating_two:
                        # continue
                        positive_data.append([self.get_score(scores, sample_one),
                                              self.get_score(scores, sample_two),
                                              0.])
                    elif rating_one > rating_two:
                        negative_data.append([self.get_score(scores, sample_one),
                                              self.get_score(scores, sample_two),
                                              1.])
                    else:
                        negative_data.append([self.get_score(scores, sample_one),
                                              self.get_score(scores, sample_two),
                                              -1.])

        shuffle(positive_data)
        shuffle(negative_data)

        length = min(len(positive_data), len(negative_data))
        positive_data = positive_data[:length]
        negative_data = negative_data[:length]

        positive_batches = []
        negative_batches = []

        for batch_n in range(len(positive_data) // self.batch_size):
            batch = positive_data[self.batch_size * batch_n:self.batch_size * (batch_n + 1)]
            positive_batches.append([tt.tensor(a) for a in zip(*batch)])

        for batch_n in range(len(negative_data) // self.batch_size):
            batch = negative_data[self.batch_size * batch_n:self.batch_size * (batch_n + 1)]
            negative_batches.append([tt.tensor(a) for a in zip(*batch)])

        batches = positive_batches + negative_batches
        shuffle(batches)

        return batches

    def _create_train(self, training: Dict[int, WarmStartUser]) -> Dict[int, Dict[str, Dict[int, int]]]:
        data = {}
        for user, warm in training.items():
            entities = list(warm.training.items())
            shuffle(entities)
            length = min(self.ask_limit, len(entities) // 2)
            ppr_train = dict(entities[:length])
            pairwise_train = dict(entities[length:])
            data[user] = {'ppr': ppr_train, 'pairwise': pairwise_train}

        return data

    def fit(self, training: Dict[int, WarmStartUser]):
        self.can_ask_about = set(self.meta.get_question_candidates(training, limit=self.ask_limit))

        # Get sentiments and entities
        sentiments = []
        for _, user in training.items():
            for entity, sentiment in user.training.items():
                self.entity_indices.add(entity)
                sentiments.append(sentiment)

        self.graphs = self.construct_graph(training)
        train = self._create_train(training)
        if self.optimal_params is None:
            best_score = -1
            best_params = None
            for parameter in _get_parameters():
                logger.debug(f'Trying with alpha: {parameter["alpha"]}')
                self._set_parameters(parameter)

                val_preds, train_preds = self._fit(training, train)
                batches = self._get_batches(train_preds, training)
                score = self.meta.validator.score(self._optimize_weights(batches, val_preds, 1), self.meta)

                # Clear cache
                self.clear_cache()

                if score > best_score:
                    logger.debug(f'New best with score: {score} and params, alpha: {parameter["alpha"]}')
                    best_score = score
                    best_params = parameter

            self.optimal_params = best_params

        # Todo insert when multiple alphas
        # self._set_parameters(self.optimal_params)
        # val_preds, train_preds = self._fit(training, train)
        # batches = self._get_batches(train_preds)
        # self._optimize_weights(batches, val_preds, 1)

    def _fit(self, train_val: Dict[int, WarmStartUser], train_pair: Dict[int, Dict[str, Dict[int, int]]]):
        val_predictions = []
        pair_predictions = []
        break_num = 100000
        i = 0
        for (val_idx, val_user), (train_idx, train_user) in tqdm(list(zip(train_val.items(), train_pair.items()))[:break_num],
                                                                 total=min(break_num, len(train_val))):
            i += 1
            val_scores = [graph.get_score(val_user.training, val_user.validation.to_list()) for graph in self.graphs]
            train_scores = [graph.get_score(train_user['ppr'], list(train_user['pairwise'].keys())) for graph in self.graphs]
            val_predictions.append((val_idx, val_user.validation, val_scores))
            pair_predictions.append((train_idx, train_user['pairwise'], train_scores))

        return val_predictions, pair_predictions

    def predict(self, items: List[int], answers: Dict[int, int]) -> Dict[int, float]:
        with tt.no_grad():
            scores = [graph.get_score(answers, items) for graph in self.graphs]
            predictions = {entity: self.model(tt.tensor(self.get_score(scores, entity)).unsqueeze(0))
                           for entity in items}

        return predictions
