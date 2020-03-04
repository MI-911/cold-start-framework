import argparse
import json
import os
import sys
from typing import List

from loguru import logger

from models.naive_recommender import NaiveRecommender
from models.shared.base_recommender import RecommenderBase
from models.shared.user import WarmStartUser

models = {
    'naive': {
        'class': NaiveRecommender
    }
}

parser = argparse.ArgumentParser()
parser.add_argument('--include', nargs='*', type=str, choices=models.keys(), help='models to include')
parser.add_argument('--exclude', nargs='*', type=str, choices=models.keys(), help='models to exclude')
parser.add_argument('--debug', action='store_true', help='enable debug mode')


def _parse_args():
    args = parser.parse_args()
    model_selection = set(models.keys()) if not args.include else set(args.include)
    if args.exclude:
        model_selection = model_selection.difference(set(args.exclude))

    if not args.debug:
        logger.remove()
        logger.add(sys.stderr, level='INFO')

    return model_selection


def _instantiate_model(model_name, meta, parameters=None):
    kwargs = {
        'meta': meta
    }

    return models[model_name]['class'](**kwargs)


def _get_parameter_path(model_name, parameter_base):
    return os.path.join(model_name, 'parameters.json')


def _get_parameters(model_name, parameter_base):
    parameter_path = _get_parameter_path(model_name, parameter_base)
    if not os.path.exists(parameter_path):
        return None

    return json.load(open(parameter_path, 'r'))


def _write_parameters(model_name, parameter_base, model: RecommenderBase):
    parameters = model.get_parameters()
    parameter_path = _get_parameter_path(model_name, parameter_base)

    if not parameters and os.path.exists(parameter_path):
        os.remove(parameter_base)
    elif parameters:
        json.dump(parameters, open(parameter_path, 'w'))


def _run_model(model_name, experiment_base, training: List[WarmStartUser], testing, meta):
    parameter_dir = os.path.join(experiment_base, model_name)
    if not os.path.exists(parameter_dir):
        os.mkdir(parameter_dir)

    model_instance = _instantiate_model(model_name, meta)
    parameters = _get_parameters(model_name, parameter_dir)
    if parameters:
        model_instance.load_parameters(parameters)

    _write_parameters(model_name, parameter_dir, model_instance)


def run():
    model_selection = _parse_args()
    print(model_selection)


if __name__ == '__main__':
    logger.info(f'Working directory: {os.getcwd()}')

    run()

