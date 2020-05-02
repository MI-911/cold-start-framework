import json
from collections import defaultdict

from tqdm import tqdm

from models.base_interviewer import InterviewerBase
from recommenders.pagerank.collaborative_pagerank_recommender import CollaborativePageRankRecommender
from recommenders.pagerank.joint_pagerank_recommender import JointPageRankRecommender
from shared.meta import Meta
from shared.utility import get_top_entities


class GreedyInterviewer(InterviewerBase):
    def __init__(self, meta: Meta, use_cuda=False):
        super().__init__(meta, use_cuda)

        self.questions = None
        self.recommender = JointPageRankRecommender(meta)
        self.recommender.parameters = {'alpha': 0.45, 'importance': {1: 0.95, 0: 0.05, -1: 0.0}}
        self.idx_uri = self.meta.get_idx_uri()

    def get_entity_name(self, idx):
        return self._get_entity_property(idx, 'name')

    def get_entity_labels(self, idx):
        return self._get_entity_property(idx, 'labels')

    def _get_entity_property(self, idx, prop):
        return self.meta.entities[self.idx_uri[idx]][prop]

    def predict(self, items, answers):
        return self.recommender.predict(items, answers)

    def interview(self, answers, max_n_questions=5):
        # Exclude answers to entities already asked about
        return self.questions

    def warmup(self, training, interview_length=5):
        self.recommender.fit(training)

        entity_scores = list()
        entities = get_top_entities(training)[:100]
        progress = tqdm(entities)

        for entity in progress:
            user_validation = list()
            for user, ratings in training.items():
                answers = {entity: ratings.training.get(entity, 0)}

                prediction = self.predict(ratings.validation.to_list(), answers)
                user_validation.append((ratings.validation, prediction))

            score = self.meta.validator.score(user_validation, self.meta)
            entity_scores.append((entity, score))

            progress.set_description(f'{self.get_entity_name(entity)}: {score}')

        label_scores = defaultdict(list)
        entity_scores = list(sorted(entity_scores, key=lambda pair: pair[1], reverse=True))

        for entity, score in entity_scores:
            primary_label = self.get_entity_labels(entity)[0]

            label_scores[primary_label].append((entities.index(entity) + 1, score))

        #rank_score = list(sorted([(entities.index(entity) + 1, score) for entity, score in entity_scores], key=lambda pair: pair[0]))
        json.dump(label_scores, open('label_scores.json', 'w'))

        self.questions = [pair[0] for pair in entity_scores]

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
        self.question_name = 'N/A'

        self.interviewer = interviewer
        self.answers = dict()

    def get_best_split(self, users, entities):




        return entity_scores[0][0]

    def construct(self, users, entities, depth=0):
        # Try splitting on each
        self.question = self.get_best_split(users, entities[:10])
        self.question_name = self.interviewer.get_entity_name(self.question)

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
