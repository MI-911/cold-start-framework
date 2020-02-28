import json

import numpy as np


class NaiveRecommender:
    def __init__(self):
        self.entity_variance = None
        self.entity_popularity = None

    def warmup(self, train):
        entity_ratings = dict()

        # Aggregate ratings per entity
        for user, data in train.items():
            for idx, sentiment in data['training'].items():
                entity_ratings.setdefault(idx, []).append(sentiment)

        # Map entities to variance and popularity
        self.entity_popularity = {idx: len(ratings) for idx, ratings in entity_ratings.items()}
        self.entity_variance = {idx: np.var(ratings) for idx, ratings in entity_ratings.items()}

        # Sort by variance * logPop
        return sorted(self.entity_variance.items(),
                      key=lambda entry: entry[1] * np.log2(self.entity_popularity[entry[0]]), reverse=True)[:10]


if __name__ == '__main__':
    training = json.load(open('../data_generation/data/training.json'))
    testing = json.load(open('../data_generation/data/testing.json'))
    meta = json.load(open('../data_generation/data/meta.json'))

    naive = NaiveRecommender()

    idx_uri = {int(v): k for k, v in meta['uri_idx'].items()}
    for item, variance in naive.warmup(training):
        print(f'{idx_uri[int(item)]}: {variance}')
