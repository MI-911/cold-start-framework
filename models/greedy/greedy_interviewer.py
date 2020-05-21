import json
import pickle
from collections import defaultdict
from math import ceil
from typing import List

from loguru import logger
from tqdm import tqdm

from models.base_interviewer import InterviewerBase
from recommenders.base_recommender import RecommenderBase
from shared.enums import Metric
from shared.meta import Meta
from shared.utility import get_top_entities
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout


def pprint_tree_as_graph(graph, node, depth, max_depth):
    node_labels = []
    node_labels.append((node, node.n_users()))

    if depth == max_depth:
        return graph, [], node_labels

    edge_labels = []

    graph.add_node(node)

    for symbol, child in [('L', node.LIKE), ('U', node.UNKNOWN), ('D', node.DISLIKE)]:

        if child:
            graph.add_edge(node, child)
            edge_labels.append(((node, child), symbol))
            graph, new_edge_labels, new_node_labels = pprint_tree_as_graph(graph, child, depth + 1, max_depth)
            edge_labels += new_edge_labels
            node_labels += new_node_labels

    return graph, edge_labels, node_labels


def pprint_tree(node, prefix="- ", label=''):
    print(prefix, label, node, sep="")

    children = [(text, child) for text, child in [('L: ', node.LIKE), ('D: ', node.DISLIKE), ('U: ', node.UNKNOWN)] if child]
    for i, (text, child) in enumerate(children):
        pprint_tree(child, f'  {prefix}', text)


class GreedyInterviewer(InterviewerBase):
    def __init__(self, meta: Meta, recommender, recommender_kwargs=None, use_cuda=False, adaptive=False):
        super().__init__(meta, use_cuda)

        self.questions = None
        self.idx_uri = self.meta.get_idx_uri()
        self.adaptive = adaptive
        self.root = None

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

        # Check if we can answer the current node
        if node.question in answers:
            next_nodes = {-1: node.DISLIKE, 0: node.UNKNOWN, 1: node.LIKE}

            return self._traverse(answers, next_nodes.get(answers.get(node.question, 0)))

        # If not, then ask about the current node
        return [node.question]

    def interview(self, answers, max_n_questions=5):
        # Follow decision tree
        if self.root:
            return self._traverse(answers, self.root)

        # Exclude answers to entities already asked about
        return self.questions

    def get_entity_scores(self, training, entities: List, existing_entities: List, metric: Metric = None):
        entity_scores = list()
        progress = tqdm(entities)

        for entity in progress:
            user_validation = list()
            for user, ratings in training.items():
                answers = {e: ratings.training.get(e, 0) for e in existing_entities + [entity]}

                prediction = self.predict(ratings.validation.to_list(), answers)
                user_validation.append((ratings.validation, prediction))

            score = self.meta.validator.score(user_validation, self.meta, metric=metric)
            entity_scores.append((entity, score))

            progress.set_description(f'{self.get_entity_name(entity)}: {score}')

        return list(sorted(entity_scores, key=lambda pair: pair[1], reverse=True))

    def _get_label_scores(self, training):
        entities = self.meta.get_question_candidates(training, limit=200)

        label_scores = defaultdict(list)
        entity_scores = self.get_entity_scores(training, entities, [], metric=Metric.NDCG)

        for entity, score in entity_scores:
            primary_label = self.get_entity_labels(entity)[0]

            label_scores[primary_label].append((entities.index(entity) + 1, score))

        json.dump(label_scores, open('label_ndcg_joint.json', 'w'))

    def _get_questions(self, training):
        questions = list()

        entities = self.meta.get_question_candidates(training, limit=100)
        for question in range(10):
            entity_scores = self.get_entity_scores(training, entities, questions)
            # return [entity for entity, _ in entity_scores if entity]

            next_question = entity_scores[0][0]
            # top_entities = [entity for entity, score in entity_scores[:ceil(len(entity_scores) * 0.05)]]
            # next_question = self.get_entity_scores(training, top_entities, questions, metric=Metric.HR)[0][0]
            questions.append(next_question)

            logger.debug(f'Question {question + 1}: {self.get_entity_name(next_question)}')

            entities = [entity for entity, _ in entity_scores if entity != next_question]

        return questions

    def warmup(self, training, interview_length=5):
        entities = self.meta.get_question_candidates(training, limit=5)
        for idx, entity in enumerate(entities):
            logger.info(f'{1 + idx}. {self.get_entity_name(entity)}')

        # self.recommender.parameters = {'alpha': 0.1, 'importance': {1: 0.95, 0: 0.05, -1: 0.0}}
        self.recommender.fit(training)

        #self._get_label_scores(training)
        #return

        if self.adaptive:
            logger.debug('Constructing adaptive interview')

            self.root = Node(self).construct(training, self.meta.get_question_candidates(training, limit=50))
            graph = nx.DiGraph()
            print(f'Constructing tree graph...')
            graph, edge_labels, node_labels = pprint_tree_as_graph(graph, self.root, 0, 4)
            edge_labels = {edge: label for edge, label in edge_labels}
            node_labels = {node: label for node, label in node_labels}

            with open('dec_tree_graph.pickle', 'wb') as fp:
                pickle.dump({
                    'graph': graph,
                    'edge_labels': edge_labels,
                    'node_labels': node_labels
                }, fp)

            print(f'Calculating positions of {graph.number_of_nodes()} nodes...')
            pos = graphviz_layout(graph)
            print(f'Drawing...')
            nx.draw(graph, pos=pos, with_labels=False, node_size=600, node_color='lightgrey', node_labels=node_labels)
            nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)
            nx.draw_networkx_labels(graph, pos, labels=node_labels)
            print(f'Showing...')
            plt.savefig('dec_tree_graph.pdf')
            plt.show()
        else:
            logger.debug('Constructing fixed-question interview')

            self.questions = self._get_questions(training)

    def get_parameters(self):
        pass

    def load_parameters(self, parameters):
        pass


def filter_users(users, entity: int, sentiments: List[int]):
    """
    Get users that have rated the specified entity with one of the specified sentiments.
    """
    return {user: ratings for user, ratings in users.items() if ratings.training.get(entity, 0) in sentiments}


def construct_rating_matrix(users, meta):
    ratings = np.zeros((len(meta.users), len(meta.entities)))
    for user, data in users.items():
        for entity, rating in data.training.items():
            ratings[user, entity] = rating

    return ratings


class Node:
    def __init__(self, interviewer, entities=None):
        self.interviewer = interviewer
        self.meta = self.interviewer.meta
        self.base_questions = entities if entities else list()

        self.LIKE = None
        self.DISLIKE = None
        self.UNKNOWN = None
        self.question = None
        self.users = {}

    def select_question(self, users, entities):
        question_scores = self.interviewer.get_entity_scores(users, entities, self.base_questions)

        return question_scores[0][0]

    def construct(self, users, entities, depth=0):
        self.users = users
        self.question = self.select_question(users, entities)
        # most_popular_entities = self.meta.get_question_candidates(users, limit=100)
        # self.question = most_popular_entities[0]

        # In the nodes below, do not consider entity split on in this parent node
        entities = [entity for entity in entities if entity != self.question]

        # Partition user groups for children
        liked_users = filter_users(users, self.question, [1])
        disliked_users = filter_users(users, self.question, [-1])
        unknown_users = filter_users(users, self.question, [0])

        base_questions = self.base_questions + [self.question]

        min_users = 1
        if depth < 5:
            if len(liked_users) >= min_users:
                self.LIKE = Node(self.interviewer, base_questions).construct(liked_users, entities, depth + 1)

            if len(disliked_users) >= min_users:
                self.DISLIKE = Node(self.interviewer, base_questions).construct(disliked_users, entities, depth + 1)

            if len(unknown_users) >= min_users:
                self.UNKNOWN = Node(self.interviewer, base_questions).construct(unknown_users, entities, depth + 1)

        return self

    def get_num_co_ratings(self):
        ratings = construct_rating_matrix(self.users, self.meta)
        n_co_ratings = 0
        for u in self.users:
            for v in self.users:
                u_rated, = np.where(ratings[u] != 0)
                v_rated, = np.where(ratings[v] != 0)
                n_co_ratings += len(set(u_rated).intersection(set(v_rated)))

        return n_co_ratings

    def n_users(self):
        return len(self.users)

    def __repr__(self):
        n_users = len(self.users)
        return f'{self.interviewer.get_entity_name(self.question)} ({n_users} users, {self.base_questions})'
