import argparse
import json
import os
import sys
import time
from collections import defaultdict
from typing import Dict, Set

import numpy as np
from loguru import logger
from tqdm import tqdm

from experiments.experiment import Dataset, Split, Experiment
from experiments.metrics import ndcg_at_k, ser_at_k, coverage, tau_at_k, hr_at_k
from models.base_interviewer import InterviewerBase
from models.configuration import models
from shared.meta import Meta
from shared.ranking import Ranking
from shared.user import ColdStartUserSet, ColdStartUser, WarmStartUser
from shared.utility import join_paths, valid_dir, get_popular_items

parser = argparse.ArgumentParser()
parser.add_argument('--input', nargs=1, type=valid_dir, help='path to input data')
parser.add_argument('--include', nargs='*', type=str, choices=models.keys(), help='models to include')
parser.add_argument('--exclude', nargs='*', type=str, choices=models.keys(), help='models to exclude')
parser.add_argument('--experiments', nargs='*', type=str, help='experiments to run')
parser.add_argument('--debug', action='store_true', help='enable debug mode')


def _instantiate_model(model_name, experiment: Experiment, meta, interview_length=-1):
    kwargs = {
        'meta': meta,
        'use_cuda': models[model_name].get('use_cuda', False)
    }

    recommender = models[model_name].get('recommender', None)
    if recommender:
        kwargs['recommender'] = recommender

    recommender_kwargs = models[model_name].get('recommender_kwargs', None)
    if recommender_kwargs:
        kwargs['recommender_kwargs'] = recommender_kwargs

    instance = models[model_name]['class'](**kwargs)
    parameters = _get_parameters(model_name, experiment, interview_length)
    if parameters:
        instance.load_parameters(parameters)

    requires_interview_length = models[model_name].get('requires_interview_length', False)

    return instance, requires_interview_length


def _get_parameter_path(parameter_base, model_name, interview_length):
    return os.path.join(parameter_base, f'parameters_{model_name}_{interview_length}q.json')


def _get_parameters(model_name, experiment: Experiment, interview_length):
    parameter_path = _get_parameter_path(experiment.path, model_name, interview_length)
    if not os.path.exists(parameter_path):
        return None

    return json.load(open(parameter_path, 'r'))


def _conduct_interview(model: InterviewerBase, answer_set: ColdStartUserSet, n_questions):
    answer_state = dict()

    for q in range(n_questions):
        try:
            next_questions = model.interview(answer_state)[:n_questions - len(answer_state)]

            for question in next_questions:
                # Specify answer as unknown if the user has no answer to it
                answer_state[question] = answer_set.answers.get(question, 0)
        except Exception as e:
            logger.error(f'Exception during interview: {e}')

        if len(answer_state) >= n_questions:
            break

    assert len(answer_state) <= n_questions

    return answer_state


def _produce_ranking(model: InterviewerBase, ranking: Ranking, answers: Dict):
    to_rank = ranking.to_list()
    item_scores = model.predict(to_rank, answers)
    try:
        # Sort items to rank by their score
        # Items not present in the item_scores dictionary default to a zero score
        return list(sorted(to_rank, key=lambda item: (item_scores.get(item, 0), item), reverse=True))
    except Exception as e:
        logger.error(f'Exception during ranking: {e}')



def _test(testing, model_instance, num_questions, upper_cutoff, meta, popular_items):
    # Keep track of answers provided
    all_answers = list()

    # Metrics
    hits = defaultdict(list)
    ndcgs = defaultdict(list)
    taus = defaultdict(list)
    sers = defaultdict(list)
    covs = defaultdict(set)

    for idx, user in tqdm(testing.items(), total=len(testing)):
        for answer_set in user.sets:
            # Query the interviewer for the next entity to ask about
            answers = _conduct_interview(model_instance, answer_set, num_questions)
            all_answers.append(answers)

            # Then produce a ranked list given the current interview state
            ranked_list = _produce_ranking(model_instance, answer_set.ranking, answers)

            # From the ranked list, get ordered binary relevance and utility
            relevance = answer_set.ranking.get_relevance(ranked_list)
            utility = answer_set.ranking.get_utility(ranked_list, meta.sentiment_utility)

            for k in range(1, upper_cutoff + 1):
                ranked_cutoff = ranked_list[:k]
                relevance_cutoff = relevance[:k]

                hits[k].append(hr_at_k(relevance, k))
                ndcgs[k].append(ndcg_at_k(utility, k))
                taus[k].append(tau_at_k(utility, k))
                sers[k].append(ser_at_k(zip(ranked_cutoff, relevance_cutoff), popular_items, k, normalize=False))
                covs[k] = covs[k].union(set(ranked_cutoff))

    return hits, ndcgs, taus, sers, covs, all_answers


def _run_model(model_name, experiment: Experiment, meta: Meta, training: Dict[int, WarmStartUser],
               testing: Dict[int, ColdStartUser], max_n_questions, upper_cutoff=50):
    model_instance, requires_interview_length = _instantiate_model(model_name, experiment, meta)

    logger.info(f'Running model {model_name}')

    # Keep track of metrics and answers at different interview lengths
    metrics = defaultdict(dict)
    all_answers = dict()

    if not requires_interview_length:
        model_instance.warmup(training)

    for num_questions in range(1, max_n_questions + 1, 1):
        logger.info(f'Conducting interviews of length {num_questions}...')

        if requires_interview_length:
            model_instance, _ = _instantiate_model(model_name, experiment, meta, num_questions)
            model_instance.warmup(training, num_questions)

        popular_items = get_popular_items(meta.recommendable_entities, training)

        hits, ndcgs, taus, sers, covs, answers = _test(testing, model_instance, num_questions, upper_cutoff, meta,
                                                       popular_items)

        hr = dict()
        ndcg = dict()
        tau = dict()
        ser = dict()
        cov = dict()

        for k in range(1, upper_cutoff + 1):
            hr[k] = np.mean(hits[k])
            ndcg[k] = np.mean(ndcgs[k])
            tau[k] = np.mean(taus[k])
            ser[k] = np.mean(sers[k])
            cov[k] = coverage(covs[k], meta.recommendable_entities)

        metrics[num_questions] = {'hr': hr, 'ndcg': ndcg, 'tau': tau, 'ser': ser, 'cov': cov}
        all_answers[num_questions] = answers

        logger.info(f'Results for {model_name}:')
        for name, value in metrics[num_questions].items():
            logger.info(f'- {name.upper()}@{meta.default_cutoff}: {value[meta.default_cutoff]}')

        yield model_instance, metrics, num_questions, all_answers


def _run_split(model_selection: Set[str], split: Split):
    training = split.data_loader.training()
    testing = split.data_loader.testing()
    meta = split.data_loader.meta()

    for model in model_selection:
        start_time = time.time()
        logger.info(f'Running {model} on {split}')

        for model_instance, metrics, length, answers in _run_model(model, split.experiment, meta, training, testing,
                                                                   max_n_questions=10):
            logger.info(f'Writing results, parameters, and answers for {model} on {split.name} with {length} questions')

            _write_answers(model, answers, split)
            _write_results(model, metrics, split)
            _write_parameters(model, split.experiment, model_instance, length)

        logger.info(f'Finished {model}, elapsed {time.time() - start_time:.2f}s')


def _write_answers(model_name, all_answers: Dict, split: Split):
    answers_dir = join_paths('results', split.experiment.name, model_name, 'answers')
    os.makedirs(answers_dir, exist_ok=True)

    # Map indices to URIs
    idx_uri = split.data_loader.meta().get_idx_uri()

    def map_pairs(pairs):
        return {idx_uri[idx]: score for idx, score in pairs.items()}

    uri_answers = {length: [map_pairs(pair) for pair in answers] for length, answers in all_answers.items()}

    with open(os.path.join(answers_dir, f'{split.name}.json'), 'w') as fp:
        json.dump(uri_answers, fp, indent=True)


def _write_results(model_name, metrics, split: Split):
    results_dir = join_paths('results', split.experiment.name, model_name)
    os.makedirs(results_dir, exist_ok=True)

    with open(os.path.join(results_dir, f'{split.name}.json'), 'w') as fp:
        json.dump(metrics, fp, indent=True)


def _write_parameters(model_name, experiment: Experiment, model: InterviewerBase, interview_length):
    parameters = model.get_parameters()
    parameter_path = _get_parameter_path(experiment.path, model_name, interview_length)

    if not parameters and os.path.exists(parameter_path):
        os.remove(parameter_path)
    elif parameters:
        json.dump(parameters, open(parameter_path, 'w'))


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

    return model_selection, args.input[0], set(args.experiments) if args.experiments else set()


def run():
    model_selection, input_path, experiments = _parse_args()

    dataset = Dataset(input_path)
    for experiment in dataset.experiments():
        if experiments and experiment.name not in experiments:
            logger.info(f'Skipping experiment {experiment}')

            continue

        for split in experiment.splits():
            _run_split(model_selection, split)


if __name__ == '__main__':
    logger.info(f'Working directory: {os.getcwd()}')

    run()
