import argparse
import itertools as it
import os
from typing import Dict

from scipy.sparse import csr_matrix


def get_combinations(parameters):
    keys, values = zip(*parameters.items())
    return [dict(zip(keys, v)) for v in it.product(*values)]


def join_paths(*paths):
    result = None

    for path in paths:
        if not result:
            result = path
        else:
            result = os.path.join(result, path)

    return result


def valid_dir(path):
    if not os.path.isdir(path):
        raise argparse.ArgumentTypeError('The specified path is not a directory')
    else:
        return path


def csr(train, sentiment_value: Dict[int, float]):
    all_ratings = []
    users = []
    items = []

    for u_idx, (user, ratings) in enumerate(train.items()):
        for entity, rating in ratings.training.items():
            value = sentiment_value.get(rating, None)
            if not value:
                continue

            all_ratings.append(value)

            users.append(user)
            items.append(entity)

    return csr_matrix((all_ratings, (users, items)))
