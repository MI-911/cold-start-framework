import argparse
import json
import multiprocessing
import os
import sys
import time
from collections import defaultdict
from concurrent.futures._base import wait
from concurrent.futures.process import ProcessPoolExecutor
from typing import Dict, Set, List

import numpy as np
from loguru import logger
from tqdm import tqdm

from experiments.experiment import Dataset, Split, Experiment
from experiments.metrics import ndcg_at_k, ser_at_k, coverage, tau_at_k, hr_at_k
from models.base_interviewer import InterviewerBase
from models.dqn.dqn_interviewer import DqnInterviewer
from models.dumb.dumb_interviewer import DumbInterviewer
from models.melu.melu_interviewer import MeLUInterviewer
from models.fmf.fmf_interviewer import FMFInterviewer
from models.lrmf.lrmf_interviewer import LRMFInterviewer
from models.naive.naive_interviewer import NaiveInterviewer
from recommenders.knn.knn_recommender import KNNRecommender
from recommenders.mf.mf_recommender import MatrixFactorizationRecommender
from recommenders.pagerank.collaborative_pagerank_recommender import CollaborativePageRankRecommender
from recommenders.pagerank.joint_pagerank_recommender import JointPageRankRecommender
from recommenders.pagerank.kg_pagerank_recommender import KnowledgeGraphPageRankRecommender
from recommenders.pagerank.linear_collaborative_pagerank_recommender import LinearCollaborativePageRankRecommender
from recommenders.pagerank.linear_joint_pagerank_recommender import LinearJointPageRankRecommender
from recommenders.pagerank.linear_kg_pagerank_recommender import LinearKGPageRankRecommender
from recommenders.random.random_recommender import RandomRecommender
from recommenders.toppop.toppop_recommender import TopPopRecommender
from shared.meta import Meta
from shared.ranking import Ranking
from shared.user import ColdStartUserSet, ColdStartUser, WarmStartUser
from shared.utility import join_paths, valid_dir

models = {
    'random': {
        'class': DumbInterviewer,
        'recommender': RandomRecommender
    },
    'top-pop': {
        'class': DumbInterviewer,
        'recommender': TopPopRecommender
    },
    'top-liked': {
        'class': DumbInterviewer,
        'recommender': TopPopRecommender,
        'recommender_kwargs': {
            'likes_only': True
        }
    },
    'lrmf': {
        'class': LRMFInterviewer,
        'requires_interview_length': True,
        'use_cuda': False
    },
    'naive-ppr-collab': {
        'class': NaiveInterviewer,
        'recommender': CollaborativePageRankRecommender
    },
    'naive-ppr-kg': {
        'class': NaiveInterviewer,
        'recommender': KnowledgeGraphPageRankRecommender
    },
    'naive-ppr-joint': {
        'class': NaiveInterviewer,
        'recommender': JointPageRankRecommender
    },
    'naive-ppr-linear-collab': {
        'class': NaiveInterviewer,
        'recommender': LinearCollaborativePageRankRecommender
    },
    'naive-ppr-linear-joint': {
        'class': NaiveInterviewer,
        'recommender': LinearJointPageRankRecommender
    },
    'naive-ppr-linear-kg': {
        'class': NaiveInterviewer,
        'recommender': LinearKGPageRankRecommender
    },
    'naive-knn': {
      'class': NaiveInterviewer,
      'recommender': KNNRecommender
    },
    'dqn-knn': {
        'class': DqnInterviewer,
        'recommender': KNNRecommender,
        'requires_interview_length': True
    },
    'naive-mf': {
        'class': NaiveInterviewer,
        'recommender': MatrixFactorizationRecommender
    },
    'dqn-mf': {
        'class': DqnInterviewer,
        'recommender': MatrixFactorizationRecommender,
        'requires_interview_length': True,
        'use_cuda': True
    },
    'dqn-ppr-kg': {
        'class': DqnInterviewer,
        'recommender': KnowledgeGraphPageRankRecommender,
        'requires_interview_length': True,
        'use_cuda': True
    },
    'dqn-ppr-collab': {
        'class': DqnInterviewer,
        'recommender': CollaborativePageRankRecommender,
        'requires_interview_length': True,
        'use_cuda': True
    },
    'dqn-ppr-joint': {
        'class': DqnInterviewer,
        'recommender': JointPageRankRecommender,
        'requires_interview_length': True,
        'use_cuda': True
    },
    'dqn-ppr-linear-kg': {
        'class': DqnInterviewer,
        'recommender': LinearKGPageRankRecommender,
        'requires_interview_length': True,
        'use_cuda': True
    },
    'dqn-ppr-linear-joint': {
        'class': DqnInterviewer,
        'recommender': LinearJointPageRankRecommender,
        'requires_interview_length': True,
        'use_cuda': True
    },
    'dqn-ppr-linear-collab': {
        'class': DqnInterviewer,
        'recommender': LinearCollaborativePageRankRecommender,
        'requires_interview_length': True,
        'use_cuda': True
    },
    'fmf': {
        'class': FMFInterviewer,
        'requires_interview_length': True,
    },
    'melu': {
        'class': MeLUInterviewer,
        'use_cuda': True
    }
}

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


def _write_parameters(model_name, experiment: Experiment, model: InterviewerBase, interview_length):
    parameters = model.get_parameters()
    parameter_path = _get_parameter_path(experiment.path, model_name, interview_length)

    if not parameters and os.path.exists(parameter_path):
        os.remove(parameter_path)
    elif parameters:
        json.dump(parameters, open(parameter_path, 'w'))


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
    try:
        to_rank = ranking.to_list()
        item_scores = model.predict(to_rank, answers)

        # Sort items to rank by their score
        # Items not present in the item_scores dictionary default to a zero score
        return list(sorted(to_rank, key=lambda item: item_scores.get(item, 0), reverse=True))
    except Exception as e:
        logger.error(f'Exception during ranking: {e}')

        return list()


def _get_popular_recents(recents: List[int], training: Dict[int, WarmStartUser]):
    recent_counts = {r: 0 for r in recents}
    for u, data in training.items():
        for idx, sentiment in data.training.items():
            if idx in recent_counts and not sentiment == 0:
                recent_counts[idx] += 1

    return [recent
            for recent, count
            in sorted(recent_counts.items(), key=lambda x: x[1], reverse=True)]


def test(testing, model_instance, num_questions, upper_cutoff, meta, popular_items):
    hits = defaultdict(list)
    ndcgs = defaultdict(list)
    taus = defaultdict(list)
    sers = defaultdict(list)
    covs = defaultdict(set)

    for idx, user in tqdm(testing.items(), total=len(testing)):
        for answer_set in user.sets:
            # Query the interviewer for the next entity to ask about
            # Then produce a ranked list given the current interview state
            answers = _conduct_interview(model_instance, answer_set, num_questions)
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

    return hits, ndcgs, taus, sers, covs


def _run_model(model_name, experiment: Experiment, meta: Meta, training: Dict[int, WarmStartUser],
               testing: Dict[int, ColdStartUser], max_n_questions, upper_cutoff=50):
    model_instance, requires_interview_length = _instantiate_model(model_name, experiment, meta)

    logger.info(f'Running model {model_name}')

    qs = defaultdict(dict)

    if not requires_interview_length:
        model_instance.warmup(training)

    for num_questions in range(1, max_n_questions + 1, 1):
        logger.info(f'Conducting interviews of length {num_questions}...')

        if requires_interview_length:
            model_instance, _ = _instantiate_model(model_name, experiment, meta, num_questions)
            model_instance.warmup(training, num_questions)

        popular_items = _get_popular_recents(meta.recommendable_entities, training)

        hits, ndcgs, taus, sers, covs = test(testing, model_instance, num_questions, upper_cutoff, meta, popular_items)

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

        qs[num_questions] = {'hr': hr, 'ndcg': ndcg, 'tau': tau, 'ser': ser, 'cov': cov}

        logger.info(f'Results for {model_name}:')
        for name, value in qs[num_questions].items():
            logger.info(f'- {name.upper()}@{meta.default_cutoff}: {value[meta.default_cutoff]}')

        yield model_instance, qs, num_questions


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

        for model_instance, qs, nq in _run_model(model, split.experiment, meta, training, testing, max_n_questions=10):
            logger.info(f'Writing results and parameters for {model} on split {split.name}, interview length {nq}')
            _write_parameters(model, split.experiment, model_instance, nq)
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
