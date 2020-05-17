import argparse
import itertools as it
import json
import os
import pickle
from functools import lru_cache, wraps
from typing import Dict
from typing import Dict, List

import numpy
from numpy import int64
from numpy.core.multiarray import ndarray
from scipy.sparse import csr_matrix

from shared.user import WarmStartUser


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


def get_top_entities(training, limit_entities=None):
    entity_ratings = dict()

    # Aggregate ratings per entity
    for user, data in training.items():
        for idx, sentiment in data.training.items():
            if limit_entities and idx not in limit_entities:
                continue

            entity_ratings.setdefault(idx, []).append(sentiment)

    return list([item[0] for item in sorted(entity_ratings.items(), key=lambda x: len(x[1]), reverse=True)])


def hashable_lru(maxsize=None):
    def inner_lru(func):
        cache = lru_cache(maxsize=maxsize)

        def deserialise(value):
            try:
                return pickle.loads(value)
            except Exception:
                return value

        def func_with_serialized_params(*args, **kwargs):
            _args = tuple([deserialise(arg) for arg in args])
            _kwargs = {k: deserialise(v) for k, v in kwargs.items()}
            return func(*_args, **_kwargs)

        cached_function = cache(func_with_serialized_params)

        @wraps(func)
        def lru_decorator(*args, **kwargs):
            _args = tuple([pickle.dumps(arg) if type(arg) in (list, dict, ndarray, int64) else arg for arg in args])
            _kwargs = {k: pickle.dumps(v) if type(v) in (list, dict, ndarray, int64) else v for k, v in kwargs.items()}
            return cached_function(*_args, **_kwargs)
        lru_decorator.cache_info = cached_function.cache_info
        lru_decorator.cache_clear = cached_function.cache_clear
        return lru_decorator

    return inner_lru
