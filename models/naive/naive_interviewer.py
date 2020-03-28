import operator

import numpy as np

from experiments.data_loader import DataLoader
from models.base_interviewer import InterviewerBase
from shared.meta import Meta


class NaiveInterviewer(InterviewerBase):
    def __init__(self, meta: Meta, recommender, use_cuda=False):
        super().__init__(meta, use_cuda)

        if not recommender:
            raise RuntimeError('No underlying recommender provided to the naive interviewer.')

        self.entity_variance = None
        self.entity_popularity = None
        self.entity_weight = None
        self.recommender = recommender(meta)

    def predict(self, items, answers):
        return self.recommender.predict(items, answers)

    def interview(self, answers, max_n_questions=5):
        # Exclude answers to entities already asked about
        valid_items = {k: v for k, v in self.entity_weight.items() if k not in answers.keys()}
        return [item[0] for item in sorted(valid_items.items(), key=operator.itemgetter(1), reverse=True)]

    @staticmethod
    def _compute_weight(popularity, variance):
        return np.log2(popularity) * variance

    def warmup(self, training, interview_length=5):
        self.recommender.fit(training)

        entity_ratings = dict()

        # Aggregate ratings per entity
        for user, data in training.items():
            for idx, sentiment in data.training.items():
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


if __name__ == '__main__':
    data_loader = DataLoader('../../data/basic/split_0')
    training = data_loader.training()
    testing = data_loader.testing()
    meta = data_loader.meta()

    naive = NaiveInterviewer(meta)
    naive.warmup(training)
    idx_uri = {int(v): k for k, v in meta.uri_idx.items()}
    entities = meta.entities
    movies = [idx for idx, movie in meta.idx_item.items() if movie]

    state = {}
    while True:
        question = int(naive.interview(state)[0])

        answer = input(f'What do you think about {entities[idx_uri[question]]["name"]}?')
        state[question] = int(answer)

        predictions = naive.predict(movies, state)
        top_movies = [pair[0] for pair in sorted(predictions.items(), key=operator.itemgetter(1), reverse=True)][:5]

        for idx, movie in enumerate(top_movies):
            print(f'{idx + 1}. {idx_uri[movie]}')
