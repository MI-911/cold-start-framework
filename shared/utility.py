import itertools as it
import os


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
