import json
from collections import defaultdict
from typing import List

from loguru import logger
from tqdm import tqdm

from models.base_interviewer import InterviewerBase
from shared.meta import Meta
from shared.utility import get_top_entities


class GreedyInterviewer(InterviewerBase):
    def __init__(self, meta: Meta, recommender, recommender_kwargs=None, use_cuda=False, recommendable_only=False):
        super().__init__(meta, use_cuda)

        self.questions = None
        self.idx_uri = self.meta.get_idx_uri()
        self.recommendable_only = recommendable_only

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

    def interview(self, answers, max_n_questions=5):
        # Exclude answers to entities already asked about
        return self.questions

    def _entity_scores(self, training, entities: List, existing_entities: List):
        entity_scores = list()
        progress = tqdm(entities)

        for entity in progress:
            user_validation = list()
            for user, ratings in training.items():
                answers = {e: ratings.training.get(e, 0) for e in existing_entities + [entity]}

                prediction = self.predict(ratings.validation.to_list(), answers)
                user_validation.append((ratings.validation, prediction))

            score = self.meta.validator.score(user_validation, self.meta)
            entity_scores.append((entity, score))

            progress.set_description(f'{self.get_entity_name(entity)}: {score}')

        return list(sorted(entity_scores, key=lambda pair: pair[1], reverse=True))

    def _get_label_scores(self, training):
        entities = get_top_entities(training)[:100]

        label_scores = defaultdict(list)
        entity_scores = self._entity_scores(training, entities, [])

        for entity, score in entity_scores:
            primary_label = self.get_entity_labels(entity)[0]

            label_scores[primary_label].append((entities.index(entity) + 1, score))

        json.dump(label_scores, open('label_scores.json', 'w'))

    def _get_questions(self, training):
        questions = list()

        limit_entities = self.meta.recommendable_entities if self.recommendable_only else None
        entities = get_top_entities(training, limit_entities)[:50]

        for _ in range(10):
            entity_scores = self._entity_scores(training, entities, questions)
            # return [entity for entity, _ in entity_scores if entity]

            next_question = entity_scores[0][0]
            entities = [entity for entity, _ in entity_scores if entity != next_question]

            questions.append(next_question)

        return questions

    def warmup(self, training, interview_length=5):
        self.recommender.fit(training)

        # self.recommender.disable_cache()

        self.questions = self._get_questions(training)

        # Print questions
        for idx, entity in enumerate(self.questions):
            logger.info(f'{idx + 1}. {self.get_entity_name(entity)}')

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
        pass

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
