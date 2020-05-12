import operator

import numpy as np

from models.base_interviewer import InterviewerBase
from shared.meta import Meta


class NaiveInterviewer(InterviewerBase):
    def __init__(self, meta: Meta, recommender=None, use_cuda=False, recommender_kwargs=None):
        super().__init__(meta, use_cuda)

        if not recommender:
            raise RuntimeError('No underlying recommender provided to the naive interviewer.')

        self.questions = None

        kwargs = {'meta': meta}
        if recommender_kwargs:
            kwargs.update(recommender_kwargs)

        self.recommender = recommender(**kwargs)

    def predict(self, items, answers):
        return self.recommender.predict(items, answers)

    def interview(self, answers, max_n_questions=5):
        # Exclude answers to entities already asked about
        return [entity for entity in self.questions if entity not in answers]

    def warmup(self, training, interview_length=5):
        self.recommender.fit(training)

        self.questions = self.meta.get_question_candidates(training)

    def get_parameters(self):
        pass

    def load_parameters(self, parameters):
        pass
