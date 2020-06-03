from collections import defaultdict
from concurrent.futures._base import wait
from concurrent.futures.process import ProcessPoolExecutor
from concurrent.futures.thread import ThreadPoolExecutor
from copy import deepcopy
from random import shuffle, choice
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
from statistics import mean, median, stdev


def _get_parameters():
    params = []

    for alpha in np.arange(0.1, 1, 0.15):
        params.append({'alpha': alpha})

    return params


class PairwiseLinear(nn.Module):
    def __init__(self, num_graphs):
        super().__init__()
        self.weights = nn.Linear(num_graphs, 1, bias=False)
        # self.weights.weight.data.uniform_(-1., 1.)
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
        self.optimizer = None
        self.margin = 0.001
        self.positive_loss_func = nn.MSELoss(reduction='mean')
        self.ranking_loss_func = nn.MarginRankingLoss(margin=self.margin, reduction='sum')
        self.triplet_loss_func = nn.MSELoss(reduction='sum')
        self.lr = 0.005
        self.beta = 0.8
        self.n_decimals = 2
        self.lr_decay = 0.96
        self.lr_decay_scheduler = None
        self.batch_size = 32

    @staticmethod
    def hinge_loss_function(x, y, t):
        x = x - y
        x = tt.pow(x, 2)
        x = tt.sub(1.0, x)
        x = tt.relu(x)
        return tt.sum(x)

    @staticmethod
    def exponential_loss(x, y, t):
        x = x - y
        x = tt.exp(-x)
        return tt.sum(x)

    @staticmethod
    def logistic_loss_function(x, y, t):
        diff = x - y
        temp = tt.exp(-diff)
        temp = tt.add(temp, 1.0)
        temp = tt.log(temp)
        return tt.sum(temp)

    @staticmethod
    def three_sample_loss(margin=0., reduction='mean'):
        def inner_func(anchor, positive, negative):
            pos_dist = tt.pow(anchor - positive, 2.)
            neg_dist = tt.pow(anchor - negative, 2.)
            res = tt.add(tt.sub(pos_dist, neg_dist), margin)
            res = tt.relu(res)
            # Use mean if reduction is not sum.
            return tt.sum(res) if reduction == 'sum' else tt.mean(res)

        return inner_func

    def construct_graph(self, training: Dict[int, WarmStartUser]) -> List[GraphWrapper]:
        raise NotImplementedError()

    def limit_weight_loss(self, limit):
        extra = self.model.weights.weight
        extra = tt.abs(extra)
        extra = tt.sub(extra, limit)
        extra = tt.relu(extra)
        extra = tt.sum(extra)
        return extra

    def _fit_triples(self, batches, predictions, epochs=100):
        logger.debug(f'Beta: {self.beta}')
        logger.debug(f'Margin: {self.margin}')
        logger.debug(f'Decimal: {self.n_decimals}')
        best_model = None
        best_score = -1
        tt.autograd.set_detect_anomaly(True)
        # t = tqdm(range(epochs), total=epochs)
        for epoch in range(epochs):
            running_loss = tt.tensor(0.)
            count = tt.tensor(0.)
            shuffle(batches)
            t = tqdm(batches)
            for batch in t:
                anchors = []
                positives = []
                negatives = []

                # Find triples
                with tt.no_grad():
                    for samples, ratings in batch:
                        samples_val = self.model(samples)
                        positive_indexes = ratings.nonzero().flatten()

                        for anchor_index in positive_indexes:
                            margin = tt.add(samples_val[anchor_index], self.margin)
                            negative_indexes = (samples_val < margin).flatten().nonzero().flatten()
                            negative_indexes = negative_indexes[(ratings[negative_indexes] == 0).nonzero().flatten()]
                            distances = tt.pow(samples_val[anchor_index] - samples_val, 2)
                            if len(negative_indexes) > 0:
                                # Find the negative sample farthest away from the postive while inside the margin.
                                negative_index = tt.argmax(distances[negative_indexes])  # TODO maybe reverse
                            else:
                                # If we did not find any, find negative min d(a, n)
                                negative_indexes = (ratings == 0).nonzero().flatten()
                                negative_index = tt.argmin(distances[negative_indexes])

                            anchors.append(samples[anchor_index].tolist())
                            negatives.append(samples[negative_indexes[negative_index]].tolist())

                self.optimizer.zero_grad()
                anchors, _, negatives = tt.tensor(anchors), tt.tensor(positives), tt.tensor(negatives)

                if len(anchors) <= 0:
                    logger.debug('skipping')
                    continue

                def temp_loss(a, p, n):
                    a_p = tt.sub(a, p).norm(dim=1).pow(2)
                    a_n = tt.sub(a, n).norm(dim=1).pow(2)

                    return tt.add(tt.sub(a_p, a_n), self.margin).relu().sum()

                anchor_scores = self.model(anchors)
                negative_scores = self.model(negatives)
                rank_loss = self.ranking_loss_func(anchor_scores, negative_scores, tt.ones(len(anchors)))
                loss = tt.mul(tt.sub(1, self.beta), self.model.weights.weight.norm()) + tt.mul(rank_loss, self.beta)

                loss.backward()
                self.optimizer.step()

                with tt.no_grad():
                    running_loss += loss
                    count += tt.tensor(1.)
                    weights = ["%.4f" % item for item in self.model.weights.weight.tolist()[0]]
                    t.set_description(f'[Epoch {epoch}] Loss: {running_loss / count:.4f}, '
                                      f'W: {weights}, BS: {best_score:.3f}')

            if epoch >= 5:
                self.lr_decay_scheduler.step()

            preds = []
            with tt.no_grad():
                current = deepcopy(self.model.weights.weight)

                # Reduce
                self.model.weights.weight.data = tt.round(self.model.weights.weight * 10 ** self.n_decimals) \
                                                 / 10 ** self.n_decimals
                for idx, val, scores in predictions:
                    p = {entity: self.model(tt.tensor(self.get_score(scores, entity)).unsqueeze(0))
                         for entity in val.to_list()}
                    preds.append((val, p))

                score = self.meta.validator.score(preds, self.meta)
                if score > best_score:
                    best_score = score
                    best_model = deepcopy(self.model.state_dict())
                    logger.debug(f'New best weight {best_model["weights.weight"]}')

                self.model.weights.weight.data = current  # Ensures model continues from where it left off.

        return best_score, best_model

    def _set_parameters(self, parameters):
        self.alpha = parameters['alpha']
        self.model = PairwiseLinear(len(self.graphs))
        self.optimizer = tt.optim.Adam(self.model.parameters(), lr=self.lr)  # , weight_decay=0.999)
        self.lr_decay_scheduler = tt.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=self.lr_decay)

        for graph in self.graphs:
            graph.alpha = self.alpha

    @staticmethod
    def get_score(scores, entity):
        return [score[entity] for score in scores]

    def _get_batches_triplets(self, preds):
        batches = []
        t = tqdm(range(256), total=256, desc='Creating data')
        for _ in t:
            shuffle(preds)
            batch = []
            # Sample from 8 users each time. We sample 8 posititives and 56 negatives
            i = 0
            while len(batch) < 16:
                idx, val, scores = preds[i]
                i += 1

                positives = [[self.get_score(scores, entity), rating] for entity, rating in val.items() if rating == 1]
                shuffle(positives)
                positives = positives

                negatives = [[self.get_score(scores, entity), 0.] for entity, rating in val.items() if rating == 0]
                if len(negatives) <= 1:
                    negatives = [[self.get_score(scores, entity), 0.] for entity, rating in val.items() if rating <= 0]

                shuffle(negatives)
                negatives = negatives

                if len(positives) <= 0 or len(negatives) <= 0:
                    continue

                batch.append([tt.tensor(a) for a in zip(*(positives + negatives))])


            batches.append(batch)

        shuffle(batches)

        return batches

    def _create_train(self, training: Dict[int, WarmStartUser]) -> Dict[int, Dict[str, Dict[int, int]]]:
        data = {}
        for user, warm in training.items():
            entities = warm.training.copy()

            # Sample one positive item.
            positives = [e for e, r in entities.items() if r == 1 and e not in self.can_ask_about]
            negatives = [e for e, r in entities.items() if r <= 0 and e not in self.can_ask_about]

            if len(positives) <= 1:
                continue

            #  Assign unseen to training
            pairwise_train = {entity: 0 for entity in warm.validation.sentiment_samples[Sentiment.UNSEEN]}

            # Add positive sample
            for pos_sample in positives + negatives:
                pairwise_train[pos_sample] = entities.pop(pos_sample)

            # Rest is used for train
            ppr_train = entities

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
            best_model = None
            for parameter in _get_parameters():
                logger.debug(f'Trying with alpha: {parameter["alpha"]}')
                self._set_parameters(parameter)

                val_preds, train_preds = self._fit(training, train)
                batches = self._get_batches_triplets(train_preds)
                score, model = self._fit_triples(batches, val_preds, 10)

                # Clear cache
                self.clear_cache()

                if score > best_score:
                    logger.debug(f'New best with score: {score} and params, alpha: {parameter["alpha"]}')
                    logger.debug(f'Weights were {model["weights.weight"]}')
                    best_score = score
                    best_params = parameter
                    best_model = model

            self.optimal_params = best_params
            self._set_parameters(self.optimal_params)
            self.model.load_state_dict(best_model)
            logger.debug(f'Using alpha {self.optimal_params["alpha"]}, and weights {self.model.weights.weight}')
        else:
            self._set_parameters(self.optimal_params)
            val_preds, train_preds = self._fit(training, train)
            batches = self._get_batches_triplets(train_preds)
            score, model = self._fit_triples(batches, val_preds, 10)
            self.model.load_state_dict(model)

        self.clear_cache()

    def _fit(self, train_val: Dict[int, WarmStartUser], train_pair: Dict[int, Dict[str, Dict[int, int]]]):
        val_predictions = []
        pair_predictions = []
        for (val_idx, val_user), (train_idx, train_user) in tqdm(zip(train_val.items(), train_pair.items()),
                                                                 total=len(train_val)):
            val_scores = [graph.get_score(val_user.training, val_user.validation.to_list()) for graph in self.graphs]
            train_scores = [graph.get_score(train_user['ppr'], list(train_user['pairwise'].keys())) for graph in
                            self.graphs]
            val_predictions.append((val_idx, val_user.validation, val_scores))
            pair_predictions.append((train_idx, train_user['pairwise'], train_scores))

        return val_predictions, pair_predictions

    def predict(self, items: List[int], answers: Dict[int, int]) -> Dict[int, float]:
        with tt.no_grad():
            scores = [graph.get_score(answers, items) for graph in self.graphs]
            predictions = {entity: self.model(tt.tensor(self.get_score(scores, entity)).unsqueeze(0))
                           for entity in items}

        return predictions
