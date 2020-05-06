import operator

import numpy as np

from models.base_interviewer import InterviewerBase
from shared.meta import Meta


class NaiveInterviewer(InterviewerBase):
    def __init__(self, meta: Meta, recommender=None, use_cuda=False, recommendable_only=False, recommender_kwargs=None):
        super().__init__(meta, use_cuda)

        if not recommender:
            raise RuntimeError('No underlying recommender provided to the naive interviewer.')

        self.entity_variance = None
        self.entity_popularity = None
        self.entity_weight = None
        self.recommendable_only = recommendable_only

        kwargs = {'meta': meta}
        if recommender_kwargs:
            kwargs.update(recommender_kwargs)

        self.recommender = recommender(**kwargs)

    def predict(self, items, answers):
        return self.recommender.predict(items, answers)

    def interview(self, answers, max_n_questions=5):
        # Exclude answers to entities already asked about
        valid_items = {k: v for k, v in self.entity_weight.items() if k not in answers.keys()}
        return [item[0] for item in sorted(valid_items.items(), key=operator.itemgetter(1), reverse=True)]

    def get_top_k_popular(self, k):
        return self.interview(dict(), max_n_questions=k)

    @staticmethod
    def _compute_weight(popularity, variance):
        return popularity

    def warmup(self, training, interview_length=5):
        self.recommender.fit(training)

        entity_ratings = dict()

        # Aggregate ratings per entity
        for user, data in training.items():
            for idx, sentiment in data.training.items():
                if self.recommendable_only and idx not in self.meta.recommendable_entities:
                    continue

                entity_ratings.setdefault(idx, []).append(sentiment)

        # Map entities to variance and popularity
        self.entity_popularity = {idx: len(ratings) for idx, ratings in entity_ratings.items()}
        self.entity_variance = {idx: np.var(ratings) for idx, ratings in entity_ratings.items()}

        # Combine variance and popularity in a joint weight
        self.entity_weight = {int(idx): self._compute_weight(self.entity_popularity[idx], self.entity_variance[idx])
                              for idx in entity_ratings.keys()}

    def get_parameters(self):
        pass

    def load_parameters(self, parameters):
        pass
