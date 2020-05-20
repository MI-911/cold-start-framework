from collections import defaultdict
from random import shuffle
from typing import List, Dict, Union

import dgl
import torch as tt
from torch.nn import functional as F
from dgl import DGLGraph, DGLHeteroGraph
from loguru import logger
from tqdm import tqdm

from recommenders.gcn.gcn import HeteroRGCN
from recommenders.base_recommender import RecommenderBase
from shared.meta import Meta
from shared.user import WarmStartUser
import numpy as np
import torch.optim as optim


def construct_relation_dict(meta: Meta):
    relation_pairs = defaultdict(list)

    for triple in meta.triples:
        h = meta.uri_idx[triple.head]
        t = meta.uri_idx[triple.tail]
        r = triple.relation

        relation_pairs[r].append((h, t))

    relations_dict = {
        ('ENTITY', r, 'ENTITY'): relation_pairs[r]
        for r in relation_pairs.keys()
    }

    u_idx = max(meta.uri_idx.values()) + 1
    return relations_dict, u_idx


def construct_rating_matrix(training: Dict[int, WarmStartUser], n_users: int, n_entities: int):
    ratings = np.zeros((n_users, n_entities))
    for u, data in training.items():
        for entity, rating in data.training.items():
            ratings[u, entity] = rating

    return ratings


class GCNRecommender(RecommenderBase):
    def __init__(self, meta: Meta):
        super().__init__(meta)
        self.directed = False
        self.batch_size = 64

        self.n_users = len(meta.users)
        self.n_entities = len(meta.entities)
        self.model: Union[HeteroRGCN, None] = None
        self.G: Union[DGLHeteroGraph, None] = None

    def fit(self, training: Dict[int, WarmStartUser]):

        relation_dict, u_idx = construct_relation_dict(self.meta)
        ratings = construct_rating_matrix(training, self.n_users, self.n_entities)
        self.G = dgl.heterograph(relation_dict)

        self.model = HeteroRGCN(self.G, in_size=10, hidden_size=10, embedding_size=10)
        optimizer = optim.Adam(self.model.parameters(), lr=0.0001)

        n_epochs = 10
        for epoch in range(n_epochs):
            users = list(training.keys())
            np.random.shuffle(users)

            epoch_loss = 0.0

            for user in tqdm(users, desc=f'[Epoch {epoch}/{n_epochs}]'):
                positive_samples, = np.where(ratings[user] == 1)
                unseen_samples, = np.where(ratings[user] == 0)
                negative_samples, = np.where(ratings[user] == -1)

                if len(positive_samples) == 0:
                    continue

                # Pick a random liked entity
                pos = np.random.choice(positive_samples)
                positive_samples = [e for e in positive_samples if not e == pos]

                # Pick a random number of the remaining ratings as input
                n_p = np.random.randint(0, len(positive_samples))
                n_n = np.random.randint(0, len(negative_samples)) if len(negative_samples) else 0

                positive_inputs = np.random.choice(positive_samples, n_p)
                negative_inputs = np.random.choice(negative_samples, n_n)

                # Pick a number of random unseen samples
                negs = np.random.choice(unseen_samples, 100)

                # Give the inputs to the model, get entity rankings.
                scores = self.model(self.G, positive_inputs, negative_inputs)
                pos_score = scores[pos]
                neg_score = scores[negs]

                loss = self.model.ranking_loss(pos_score, neg_score)

                self.model.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.detach().cpu().numpy()

            logger.info(f'Loss: {epoch_loss / self.n_users}')

    def predict(self, items: List[int], answers: Dict[int, int]) -> Dict[int, float]:
        liked_entities = [e for e, a in answers.items() if a == 1]
        disliked_entities = [e for e, a in answers.items() if a == -1]

        scores = self.model(self.G, liked_entities, disliked_entities)
        return {e: scores[e] for e in items}

    def clear_cache(self):
        pass