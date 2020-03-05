import argparse
import json
import operator
import os
import sys
import time
from typing import Dict, Set

from loguru import logger

from experiments.experiment import Dataset, Split, Experiment
from models.naive_recommender import NaiveRecommender
from models.shared.base_recommender import RecommenderBase
from models.shared.meta import Meta
from models.shared.user import WarmStartUser, ColdStartUser, ColdStartUserSet
from shared.validators import valid_dir

models = {
    'naive': {
        'class': NaiveRecommender
    }
}

parser = argparse.ArgumentParser()
parser.add_argument('--input', nargs=1, type=valid_dir, help='path to input data')
parser.add_argument('--include', nargs='*', type=str, choices=models.keys(), help='models to include')
parser.add_argument('--exclude', nargs='*', type=str, choices=models.keys(), help='models to exclude')
parser.add_argument('--debug', action='store_true', help='enable debug mode')


def _instantiate_model(model_name, experiment: Experiment, meta):
    kwargs = {
        'meta': meta
    }

    instance = models[model_name]['class'](**kwargs)
    parameters = _get_parameters(model_name, experiment)
    if parameters:
        instance.load_parameters(parameters)

    return instance


def _get_parameter_path(parameter_base, model_name):
    return os.path.join(parameter_base, f'parameters_{model_name}.json')


def _get_parameters(model_name, experiment: Experiment):
    parameter_path = _get_parameter_path(experiment.path, model_name)
    if not os.path.exists(parameter_path):
        return None

    return json.load(open(parameter_path, 'r'))


def _write_parameters(model_name, experiment: Experiment, model: RecommenderBase):
    parameters = model.get_parameters()
    parameter_path = _get_parameter_path(experiment.path, model_name)

    if not parameters and os.path.exists(parameter_path):
        os.remove(parameter_path)
    elif parameters:
        json.dump(parameters, open(parameter_path, 'w'))


def _conduct_interview(model: RecommenderBase, answer_set: ColdStartUserSet, n_questions=5):
    answer_state = dict()

    while len(answer_state) < n_questions:
        next_questions = model.interview(answer_state)[:n_questions - len(answer_state)]

        for question in next_questions:
            # Specify answer as unknown if the user has no answer to it
            answer_state[question] = answer_set.answers.get(question, 0)

    assert len(answer_state) <= n_questions

    return answer_state


def _produce_ranking(model: RecommenderBase, answer_set: ColdStartUserSet, answers: Dict):
    to_rank = [answer_set.positive] + answer_set.negative
    item_scores = sorted(model.predict(to_rank, answers).items(), key=operator.itemgetter(1), reverse=True)

    return [item[0] for item in item_scores]


def _run_model(model_name, experiment: Experiment, meta: Meta, training: Dict[int, WarmStartUser],
               testing: Dict[int, ColdStartUser]):
    model_instance = _instantiate_model(model_name, experiment, meta)
    model_instance.warmup(training)

    splits, hits = 0, 0
    for idx, user in testing.items():
        for answer_set in user.sets:
            answers = _conduct_interview(model_instance, answer_set)
            ranking = _produce_ranking(model_instance, answer_set, answers)

            if answer_set.positive in ranking[:10]:
                hits += 1
            splits += 1

    logger.info(f'{hits / splits * 100:.2f}% HR@10')

    _write_parameters(model_name, experiment, model_instance)


def _run_split(model_selection: Set[str], split: Split):
    training = split.data_loader.training()
    testing = split.data_loader.testing()
    meta = split.data_loader.meta()

    for model in model_selection:
        start_time = time.time()
        logger.info(f'Running {model} on {split}')

        _run_model(model, split.experiment, meta, training, testing)

        logger.info(f'Finished {model}, elapsed {time.time() - start_time:.2f}s')


def _parse_args():
    args = parser.parse_args()
    model_selection = set(models.keys()) if not args.include else set(args.include)
    if args.exclude:
        model_selection = model_selection.difference(set(args.exclude))

    if not args.debug:
        logger.remove()
        logger.add(sys.stderr, level='INFO')

    return model_selection, args.input[0]


def run():
    model_selection, input_path = _parse_args()

    dataset = Dataset(input_path)
    for experiment in dataset.experiments():
        for split in experiment.splits():
            _run_split(model_selection, split)


if __name__ == '__main__':
    logger.info(f'Working directory: {os.getcwd()}')

    run()
