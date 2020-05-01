import json
import operator
from typing import Dict

import numpy as np
from loguru import logger
from tqdm import tqdm

from models.base_interviewer import InterviewerBase
from recommenders.pagerank.joint_pagerank_recommender import JointPageRankRecommender
from recommenders.pagerank.kg_pagerank_recommender import KnowledgeGraphPageRankRecommender
from shared.enums import Metric
from shared.meta import Meta


def get_top_entities(training):
    entity_ratings = dict()

    # Aggregate ratings per entity
    for user, data in training.items():
        for idx, sentiment in data.training.items():
            entity_ratings.setdefault(idx, []).append(sentiment)

    return list([item[0] for item in sorted(entity_ratings.items(), key=lambda x: len(x[1]), reverse=True)])


class GreedyInterviewer(InterviewerBase):
    def __init__(self, meta: Meta, use_cuda=False):
        super().__init__(meta, use_cuda)

        self.questions = None
        self.recommender = JointPageRankRecommender(meta)
        self.recommender.optimal_params = {'alpha': 0.5, 'importance': {1: 0.95, 0: 0.05, -1: 0.0}}
        self.idx_uri = self.meta.get_idx_uri()

    def get_idx_name(self, idx):
        return self.meta.entities[self.idx_uri[idx]]['name']

    def predict(self, items, answers):
        return self.recommender.predict(items, answers)

    def interview(self, answers, max_n_questions=5):
        # Exclude answers to entities already asked about
        return self.questions

    def get_top_k_popular(self, k):
        return self.interview(dict(), max_n_questions=k)

    @staticmethod
    def _compute_weight(popularity, variance):
        return popularity

    def warmup(self, training, interview_length=5):
        self.recommender.fit(training)

        root = Node(self).construct(training, get_top_entities(training))
        print('done')

    def get_parameters(self):
        pass

    def load_parameters(self, parameters):
        pass


def remove_users(users, entity, sentiment):
    new_users = dict()

    for user, ratings in users.items():
        if ratings.training.get(entity, 0) != sentiment:
            new_users[user] = ratings

    return new_users


class Node:
    def __init__(self, interviewer):
        self.LIKE = None
        self.DISLIKE = None
        self.UNKNOWN = None

        self.question = None
        self.interviewer = interviewer

    def get_best_split(self, users, entities):
        entity_scores = list()
        progress = tqdm(entities)

        for entity in progress:
            user_validation = list()
            for user, ratings in users.items():
                answers = {entity: ratings.training.get(entity, 0)}

                prediction = self.interviewer.predict(ratings.validation.to_list(), answers)
                user_validation.append((ratings.validation, prediction))

            score = self.interviewer.meta.validator.score(user_validation, self.interviewer.meta)
            entity_scores.append((entity, score))

            progress.set_description(f'{self.interviewer.get_idx_name(entity)}: {score}')

        entity_scores = list(sorted(entity_scores, key=lambda pair: pair[1], reverse=True))

        return entity_scores[0][0]

    def construct(self, users, entities, depth=0):
        # Try splitting on each
        self.question = self.get_best_split(users, entities[:10])

        # In the nodes below, do not consider entity split on in this parent node
        entities = [entity for entity in entities if entity != self.question]

        liked_users = remove_users(users, self.question, -1)
        disliked_users = remove_users(users, self.question, 1)
        unknown_users = users

        if depth < 4:
            self.LIKE = Node(self.interviewer).construct(liked_users, entities, depth + 1)
            self.DISLIKE = Node(self.interviewer).construct(disliked_users, entities, depth + 1)
            self.UNKNOWN = Node(self.interviewer).construct(unknown_users, entities, depth + 1)

        return self
