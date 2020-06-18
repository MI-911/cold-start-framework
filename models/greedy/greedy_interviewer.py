import json
import pickle
import time
from collections import defaultdict
from concurrent.futures.process import ProcessPoolExecutor
from concurrent.futures.thread import ThreadPoolExecutor
from math import ceil
from typing import List

from loguru import logger
from tqdm import tqdm

from models.base_interviewer import InterviewerBase
from partitioners.partition_interview import _chunks
from recommenders.base_recommender import RecommenderBase
from shared.enums import Metric
from shared.meta import Meta


def pprint_tree(node, prefix="- ", label=''):
    print(prefix, label, node, sep="")

    children = [(text, child) for text, child in [('L: ', node.LIKE), ('D: ', node.DISLIKE)] if child]
    for i, (text, child) in enumerate(children):
        pprint_tree(child, f'  {prefix}', text)


class GreedyInterviewer(InterviewerBase):
    def __init__(self, meta: Meta, recommender, recommender_kwargs=None, use_cuda=False, adaptive=False,
                 cov_fraction=None):
        super().__init__(meta, use_cuda)

        self.questions = None
        self.idx_uri = self.meta.get_idx_uri()
        self.adaptive = adaptive
        self.root = None
        # self.meta.validator.metric = Metric.HR
        self.cov_fraction = cov_fraction
        logger.info(f'Cov fraction: {self.cov_fraction}')

        if isinstance(recommender, RecommenderBase):
            self.recommender = recommender
        else:
            kwargs = {'meta': meta}
            if recommender_kwargs:
                kwargs.update(recommender_kwargs)

            self.recommender = recommender(**kwargs)

    def get_entity_name(self, idx):
        return self._get_entity_property(idx, 'name')

    def get_entity_labels(self, idx):
        return self._get_entity_property(idx, 'labels')

    def _get_entity_property(self, idx, prop):
        return self.meta.entities[self.idx_uri[idx]][prop]

    def predict(self, items, answers):
        return self.recommender.predict(items, answers)

    def _traverse(self, answers, node):
        # Check if node is empty, i.e. nothing to ask about
        if not node:
            return []

        # If there are no answers left, just return these questions
        # so we can continue
        if len(answers) == 0:
            return node.questions

        # If the node has fixed questions, we don't need to traverse further
        # since it's a leaf that stopped growing due to too few users
        if node.is_fixed():
            return node.questions

        # Otherwise, check the answers for this question and determine
        # the appropriate child node to go to
        question = node.questions[0]
        if question in answers:
            traverse_to = {-1: node.DISLIKE, 0: node.UNKNOWN, 1: node.LIKE}
            remaining_answers = {e: a for e, a in answers.items() if not e == question}
            return self._traverse(remaining_answers, traverse_to.get(answers.get(question, 0)))

        # As a final catch all, just return this node's questions
        return node.questions

    def interview(self, answers, max_n_questions=5):
        # Follow decision tree
        if self.root:
            return self._traverse(answers, self.root)

        # Exclude answers to entities already asked about
        return self.questions

    def _get_entity_scores(self, training, entities: List, existing_entities: List, metric: Metric = None, cutoff=None):
        entity_scores = list()

        for entity in entities:
            user_validation = list()
            for user, ratings in training.items():
                answers = {e: ratings.training.get(e, 0) for e in existing_entities + [entity]}

                prediction = self.predict(ratings.validation.to_list(), answers)
                user_validation.append((ratings.validation, prediction))

            score = self.meta.validator.score(user_validation, self.meta, metric=metric, cutoff=cutoff)
            entity_scores.append((entity, score))

        return entity_scores

    def get_entity_scores(self, training, entities: List, existing_entities: List, metric: Metric = None, cutoff=None):
        entity_scores = list()

        now = time.time()

        futures = list()
        with ProcessPoolExecutor(max_workers=3) as e:
            for chunk in _chunks(entities, 1 / 3):
                futures.append(e.submit(self._get_entity_scores, training, chunk, existing_entities, metric, cutoff))

        for future in futures:
            entity_scores.extend(future.result())
        took_time = time.time() - now

        logger.info(f'Took {took_time:.2f}s, {len(training)} users, {len(entity_scores) / took_time:.2f} it/s, {(len(entity_scores) * len(training)) / took_time:.2f} rec/s')

        return list(sorted(entity_scores, key=lambda pair: pair[1], reverse=True))

    def _get_label_scores(self, training):
        entities = self.meta.get_question_candidates(training, limit=200)

        label_scores = defaultdict(list)
        entity_scores = self.get_entity_scores(training, entities, [], metric=Metric.NDCG)

        for entity, score in entity_scores:
            primary_label = self.get_entity_labels(entity)[0]

            label_scores[primary_label].append((entities.index(entity) + 1, score))

        json.dump(label_scores, open('label_ndcg_joint.json', 'w'))

    def get_questions(self, training, num_questions=10, entities=None, base_questions=None):
        questions = list()

        entities = entities if entities else self.meta.get_question_candidates(training, limit=100)
        base_questions = base_questions if base_questions else list()

        for question in range(num_questions):
            entity_scores = self.get_entity_scores(training, entities, questions + base_questions, metric=Metric.MIXED)

            if self.cov_fraction:
                top_entities = [entity for entity, score in entity_scores]
                top_entities = top_entities[:max(1, ceil(len(top_entities) * self.cov_fraction))]

                entity_scores = self.get_entity_scores(training, top_entities, questions, metric=Metric.MIXED)

            next_question = entity_scores[0][0]

            # top_entities = [entity for entity, score in entity_scores[:ceil(len(entity_scores) * 0.05)]]
            # next_question = self.get_entity_scores(training, top_entities, questions, metric=Metric.HR)[0][0]
            questions.append(next_question)

            logger.debug(f'Question {question + 1}: {self.get_entity_name(next_question)}')

            entities = [entity for entity in entities if entity != next_question]

        return questions

    def warmup(self, training, interview_length=5):
        # self.recommender.parameters = {'alpha': 0.2, 'importance': {1: 0.95, 0: 0.05, -1: 0.0}}
        self.recommender.fit(training)

        if self.adaptive:
            logger.debug(f'Constructing {interview_length}-length adaptive interview')

            self.root = Node(self).construct(
                training, self.meta.get_question_candidates(training, limit=100), max_depth=interview_length - 1)
            pprint_tree(self.root)
        else:
            logger.debug('Constructing fixed-question interview')

            self.questions = self.get_questions(training, interview_length)

    def get_parameters(self):
        pass

    def load_parameters(self, parameters):
        pass


def filter_users(users, entity: int, sentiments: List[int]):
    """
    Get users that have rated the specified entity with one of the specified sentiments.
    """
    return {user: ratings for user, ratings in users.items() if ratings.training.get(entity, 0) in sentiments}


class Node:
    def __init__(self, interviewer: GreedyInterviewer, entities=None):
        self.interviewer = interviewer
        self.base_questions = entities if entities else list()

        self.LIKE = None
        self.DISLIKE = None
        self.UNKNOWN = None
        self.questions = None
        self.users = []
        self.depth = 0

    def select_question(self, users, entities):
        return self.interviewer.get_questions(users, num_questions=1, base_questions=self.base_questions,
                                              entities=entities)[0]

    def get_popular_questions(self, users):
        questions = self.interviewer.meta.get_question_candidates(users)

        return [question for question in questions if question not in self.base_questions]

    def get_fixed_questions(self, users):
        # Consider the top-100 locally most popular entities
        entities = self.get_popular_questions(users)[:200]
        entity_scores = self.interviewer.get_entity_scores(users, entities, self.base_questions)

        return [entity for entity, score in entity_scores]

    def construct(self, users, entities, max_depth, depth=0):
        min_users = 30
        self.depth = depth

        # self.interviewer.recommender.update_graph(users)

        # If this node doesn't have enough users to warrant a node split, we
        # can just assign the remaining interview questions as fixed questions
        if len(users) < min_users or depth > 9:
            # Use GreedyExtend to get remaining fixed questions
            # self.questions = self.get_popular_questions(users)[:max_depth - (depth - 1)]
            self.questions = self.get_fixed_questions(users)[:max_depth - (depth - 1)]

            return self

        # We have enough users to split, so continue
        self.questions = [self.select_question(users, entities)]

        # Don't split if this is the last question of the interview
        if depth == max_depth:
            return self

        # We should split the users - first remove this question from the
        # candidate questions
        entities = [entity for entity in entities if entity not in self.questions]

        # Partition user groups for children
        question = self.questions[0]

        liked_users = filter_users(users, question, [1])
        disliked_users = filter_users(users, question, [-1])

        base_questions = self.base_questions + self.questions

        self.LIKE = Node(self.interviewer, base_questions).construct(liked_users, entities, max_depth, depth + 1)
        self.DISLIKE = Node(self.interviewer, base_questions).construct(disliked_users, entities, max_depth, depth + 1)
        self.UNKNOWN = Node(self.interviewer, base_questions).construct(users, entities, max_depth, depth + 1)

        return self

    def is_fixed(self):
        return len(self.questions) > 1

    def __repr__(self):
        return str([self.interviewer.get_entity_name(question) for question in self.questions])
