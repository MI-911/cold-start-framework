import pickle
from typing import Dict, List, Union

import numpy as np
from loguru import logger
from tqdm import tqdm

from models.base_interviewer import InterviewerBase
from models.lrmf.lrmf import LIKE, DISLIKE, LRMF, Tree
from shared.meta import Meta
from shared.user import WarmStartUser
from shared.utility import get_combinations


def visualise_tree(tree: Tree, meta: Meta):
    reverse_entity_map = {idx: uri for uri, idx in meta.uri_idx.items()}

    def _idx_to_name(idx: int):
        uri = reverse_entity_map[idx]
        return meta.entities[uri]['name']

    indentation = ''.join(['|-' for _ in range(tree.depth)])

    if tree.is_leaf():
        questions = ', '.join([_idx_to_name(q) for q in tree.l2_questions])
    else:
        questions = _idx_to_name(tree.question)

    print(f'{indentation}-> {questions}')
    if not tree.is_leaf():
        visualise_tree(tree.children[LIKE], meta)
        visualise_tree(tree.children[DISLIKE], meta)


RATING_MAP = {
    1: LIKE,
    0: DISLIKE,
    -1: DISLIKE
}


def get_rating_matrix(training, n_users, n_entities):
    """
    Returns an [n_users x n_entities] ratings matrix.
    """

    R = np.ones((n_users, n_entities)) * RATING_MAP[0]
    for user, data in training.items():
        for entity, rating in data.training.items():
            R[user, entity] = RATING_MAP[rating]

    return R


def choose_candidates(rating_matrix, n=100):
    """
    Selects n candidates that can be asked towards in an interview.
    """
    # TODO: Choose candidate items with a mix between popularity and diversity
    n_ratings = np.zeros(rating_matrix.shape[1])
    for i, item_column in enumerate(rating_matrix.T):
        for rating in item_column:
            if rating == LIKE:
                n_ratings[i] += 1

    n_ratings = sorted([(entity, rs) for entity, rs in enumerate(n_ratings)], key=lambda x: x[1], reverse=True)
    return [entity for entity, rs in n_ratings][:n]


class LRMFInterviewer(InterviewerBase):
    def __init__(self, meta: Meta, use_cuda=False):
        super(LRMFInterviewer, self).__init__(meta, use_cuda)
        self.meta = meta
        self.model: LRMF = Union[LRMF, None]

        self.n_candidates = 100

        self.n_users = len(meta.users)
        self.n_entities = len(meta.entities)

        self.params = None
        self.best_model = None
        self.best_score = 0

    def warmup(self, training: Dict[int, WarmStartUser], interview_length=5):
        self.best_model = None
        self.best_score = 0

        # Pseudo-evenly split the number of global and local questions
        l1 = interview_length // 2
        l2 = interview_length - l1

        if not self.params:
            param_scores = []
            for params in get_combinations({'reg': [0.01]}):
                logger.info(f'Fitting LRMF with params {params}')
                self.model = LRMF(n_users=self.n_users, n_entities=self.n_entities,
                                  l1=l1, l2=l2, kk=-1,  # See notes
                                  regularisation=params['reg'])

                best_score = self._fit(training)
                param_scores.append((params, best_score))

            best_params, _ = list(sorted(param_scores, key=lambda x: x[1], reverse=True))[0]
            logger.info(f'Found best params for LRMF: {best_params}')
            self.model = LRMF(n_users=self.n_users, n_entities=self.n_entities,
                              l1=l1, l2=l2, kk=-1,  # See notes
                              regularisation=best_params['reg'])
            self._fit(training)

        else:
            logger.info(f'Reusing parameters for LRMF: {self.params}')
            self.model = LRMF(n_users=self.n_users, n_entities=self.n_entities,
                              l1=l1, l2=l2, kk=-1,  # See notes
                              regularisation=self.params['reg'])
            self._fit(training)

    def _fit(self, users: Dict[int, WarmStartUser], n_iterations=5) -> float:
        self.best_score = 0
        R = get_rating_matrix(users, self.n_users, self.n_entities)
        candidates = choose_candidates(R, n=100)

        for iteration in tqdm(range(n_iterations), desc=f'[Training LRMF]'):
            self.model.fit(R, candidates)
            score = self._validate(users)

            if score > self.best_score:
                self.best_score = score
                self.best_model = pickle.loads(pickle.dumps(self.model))

        self.model = self.best_model
        return self.best_score

    def _validate(self, users: Dict[int, WarmStartUser]):
        predictions = []
        for u_idx, user in users.items():
            prediction = self.model.validate(u_idx, user.validation.to_list())
            predictions.append((user.validation, prediction))

        return self.meta.validator.score(predictions, self.meta)

    def interview(self, answers: Dict[int, int], max_n_questions=5) -> List[int]:
        return [self.model.interview({e: RATING_MAP[a] for e, a in answers.items()})]

    def predict(self, items: List[int], answers: Dict[int, int]) -> Dict[int, float]:
        return self.model.rank(items, {e: RATING_MAP[a] for e, a in answers.items()})

    def get_parameters(self):
        return self.params

    def load_parameters(self, params):
        self.params = params
