import argparse
import json
import operator
import os
import sys
import time
from collections import defaultdict
from typing import Dict, Set, List

import numpy as np
from loguru import logger
from tqdm import tqdm

from experiments.experiment import Dataset, Split, Experiment
from experiments.metrics import ndcg_at_k, ser_at_k, coverage
from models.fmf.fmf_recommender import FMFRecommender
from models.naive.naive_recommender import NaiveRecommender
from models.naive.mf.mf_recommender import MatrixFactorisationRecommender
from models.shared.base_recommender import RecommenderBase
from models.shared.meta import Meta
from models.shared.user import WarmStartUser, ColdStartUser, ColdStartUserSet
from shared.utility import join_paths
from shared.validators import valid_dir

models = {
    'naive': {
        'class': NaiveRecommender,
        'requires_interview_length': False,
        'use_cuda': False
    },
    'fmf': {
        'class': FMFRecommender,
        'requires_interview_length': True,
        'use_cuda': False
    },
    'mf': {
        'class': MatrixFactorisationRecommender,
        'requires_interview_length': False,
        'use_cuda': False
    }
}

parser = argparse.ArgumentParser()
parser.add_argument('--input', nargs=1, type=valid_dir, help='path to input data')
parser.add_argument('--include', nargs='*', type=str, choices=models.keys(), help='models to include')
parser.add_argument('--exclude', nargs='*', type=str, choices=models.keys(), help='models to exclude')
parser.add_argument('--debug', action='store_true', help='enable debug mode')


def _instantiate_model(model_name, experiment: Experiment, meta, interview_length=-1):
    kwargs = {
        'meta': meta,
        'use_cuda': models[model_name]['use_cuda']
    }

    instance = models[model_name]['class'](**kwargs)
    parameters = _get_parameters(model_name, experiment, interview_length)
    if parameters:
        instance.load_parameters(parameters)

    requires_interview_length = models[model_name]['requires_interview_length']

    return instance, requires_interview_length


def _get_parameter_path(parameter_base, model_name, interview_length):
    return os.path.join(parameter_base, f'parameters_{model_name}_{interview_length}q.json')


def _get_parameters(model_name, experiment: Experiment, interview_length):
    parameter_path = _get_parameter_path(experiment.path, model_name, interview_length)
    if not os.path.exists(parameter_path):
        return None

    return json.load(open(parameter_path, 'r'))


def _write_parameters(model_name, experiment: Experiment, model: RecommenderBase, interview_length):
    parameters = model.get_parameters()
    parameter_path = _get_parameter_path(experiment.path, model_name, interview_length)

    if not parameters and os.path.exists(parameter_path):
        os.remove(parameter_path)
    elif parameters:
        json.dump(parameters, open(parameter_path, 'w'))


def _conduct_interview(model: RecommenderBase, answer_set: ColdStartUserSet, n_questions=5):
    answer_state = dict()

    for q in range(n_questions):
        next_questions = model.interview(answer_state)[:n_questions - len(answer_state)]

        for question in next_questions:
            # Specify answer as unknown if the user has no answer to it
            answer_state[question] = answer_set.answers.get(question, 0)

        if len(answer_state) >= n_questions:
            break

    assert len(answer_state) <= n_questions

    return answer_state


def _produce_ranking(model: RecommenderBase, answer_set: ColdStartUserSet, answers: Dict):
    to_rank = [answer_set.positive] + answer_set.negative
    item_scores = sorted(model.predict(to_rank, answers).items(), key=operator.itemgetter(1), reverse=True)

    return [item[0] for item in item_scores]


def _get_relevance_list(ranking, positive_item):
    return [int(item == positive_item) for item in ranking]


def _get_popular_recents(recents: List[int], training: Dict[int, WarmStartUser]):
    recent_counts = {r: 0 for r in recents}
    for u, data in training.items():
        for idx, sentiment in data.training.items():
            if idx in recent_counts and not sentiment == 0:
                recent_counts[idx] += 1

    return [recent
            for recent, count
            in sorted(recent_counts.items(), key=lambda x: x[1], reverse=True)]


def _run_model(model_name, experiment: Experiment, meta: Meta, training: Dict[int, WarmStartUser],
               testing: Dict[int, ColdStartUser], max_n_questions=5, upper_cutoff=50):
    model_instance, requires_interview_length = _instantiate_model(model_name, experiment, meta)

    qs = defaultdict(list)

    if not requires_interview_length:
        model_instance.warmup(training)

    for nq in range(1, max_n_questions + 1, 1):
        hits = defaultdict(list)
        ndcgs = defaultdict(list)
        sers = defaultdict(list)
        covs = defaultdict(set)

        if requires_interview_length:
            model_instance, _ = _instantiate_model(model_name, experiment, meta, nq)
            model_instance.warmup(training, nq)

        popular_items = _get_popular_recents(meta.recommendable_entities, training)

        for idx, user in tqdm(testing.items(), desc='[Testing]'):
            for answer_set in user.sets:
                answers = _conduct_interview(model_instance, answer_set)
                ranking = _produce_ranking(model_instance, answer_set, answers)
                relevance = _get_relevance_list(ranking, answer_set.positive)

                for k in range(1, upper_cutoff + 1):
                    cutoff = relevance[:k]

                    hits[k].append(1 in cutoff)
                    ndcgs[k].append(ndcg_at_k(cutoff, k))
                    sers[k].append(ser_at_k(zip(ranking[:k], cutoff), popular_items, k, normalize=False))
                    covs[k] = covs[k].union(set(ranking[:k]))

        hr = dict()
        ndcg = dict()
        ser = dict()
        cov = dict()

        for k in range(1, upper_cutoff + 1):
            hr[k] = np.mean(hits[k])
            ndcg[k] = np.mean(ndcgs[k])
            ser[k] = np.mean(sers[k])
            cov[k] = coverage(covs[k], meta.recommendable_entities)

        _write_parameters(model_name, experiment, model_instance, nq)
        qs[nq] = [hr, ndcg, ser, cov]

    return qs


def _write_results(model_name, qs, split: Split):
    results_dir = join_paths('results', split.experiment.name, model_name)
    os.makedirs(results_dir, exist_ok=True)

    with open(os.path.join(results_dir, f'{split.name}.json'), 'w') as fp:
        json.dump(qs, fp, indent=True)


def _run_split(model_selection: Set[str], split: Split):
    training = split.data_loader.training()
    testing = split.data_loader.testing()
    meta = split.data_loader.meta()

    for model in model_selection:
        start_time = time.time()
        logger.info(f'Running {model} on {split}')

        qs = _run_model(model, split.experiment, meta, training, testing, max_n_questions=2)
        _write_results(model, qs, split)

        logger.info(f'Finished {model}, elapsed {time.time() - start_time:.2f}s')


def _parse_args():
    args = parser.parse_args()
    model_selection = set(models.keys()) if not args.include else set(args.include)
    if args.exclude:
        model_selection = model_selection.difference(set(args.exclude))

    if not args.debug:
        logger.remove()
        logger.add(sys.stderr, level='INFO')

    if not args.input:
        args.input = ['../data']

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
