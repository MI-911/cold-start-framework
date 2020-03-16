import numpy as np
import operator

from experiments.data_loader import DataLoader
from models.shared.base_recommender import RecommenderBase
from models.shared.meta import Meta


class NaiveRecommender(RecommenderBase):
    def __init__(self, meta: Meta):
        super().__init__(meta)

        self.entity_variance = None
        self.entity_popularity = None
        self.entity_weight = None

    def predict(self, items, answers):
        from random import randint
        return {item: randint(0, 1000) for item in items}
        # raise NotImplementedError()

    def interview(self, answers, max_n_questions=5):
        # Exclude answers to entities already asked about
        valid_items = {k: v for k, v in self.entity_weight.items() if k not in answers.keys()}
        return [item[0] for item in sorted(valid_items.items(), key=operator.itemgetter(1), reverse=True)]

    @staticmethod
    def _compute_weight(popularity, variance):
        return np.log2(popularity) * variance

    def warmup(self, training):
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
    data_loader = DataLoader('../data/basic/data')
    training = data_loader.training()
    testing = data_loader.testing()
    meta = data_loader.meta()

    naive = NaiveRecommender(meta)
    naive.warmup(training)
    idx_uri = {int(v): k for k, v in meta.uri_idx.items()}
    entities = meta.entities

    state = {}
    while True:
        question = int(naive.interview(state)[0])

        answer = input(f'What do you think about {entities[idx_uri[question]]["name"]}?')
        state[question] = int(answer)