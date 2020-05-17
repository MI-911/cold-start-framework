import pickle
from typing import Dict, List, Union

from tqdm import tqdm

from models.base_interviewer import InterviewerBase
from models.fmf.fmf import LIKE, DISLIKE, UNKNOWN, FMF, Tree
import numpy as np
from loguru import logger

from shared.meta import Meta
from shared.user import WarmStartUser
from shared.utility import get_combinations


def visualise_tree(tree: Tree, meta: Meta):
    reverse_entity_map = {idx: uri for uri, idx in meta.uri_idx.items()}

    def _idx_to_name(idx: int):
        uri = reverse_entity_map[idx]
        return meta.entities[uri]['name']

    indentation = ''.join(['|-' for _ in range(tree.depth)])

    questions = _idx_to_name(tree.question) if not tree.is_leaf() else "(Make recommendation)"

    print(f'{indentation}|-> {questions}')
    if not tree.is_leaf():
        visualise_tree(tree.children[LIKE], meta)
        visualise_tree(tree.children[DISLIKE], meta)


RATING_MAP = {
    1: LIKE,
    0: UNKNOWN,
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
    n_ratings = rating_matrix.sum(axis=0)
    n_ratings = sorted([(entity, rs) for entity, rs in enumerate(n_ratings)], key=lambda x: x[1], reverse=True)
    return [entity for entity, rs in n_ratings][:n]


class FMFInterviewer(InterviewerBase):
    def __init__(self, meta, use_cuda=False):
        super(FMFInterviewer, self).__init__(meta, use_cuda)
        self.meta = meta
        self.model: FMF = Union[FMF, None]

        self.n_candidates = 100

        self.n_users = len(meta.users)
        self.n_entities = len(meta.entities)

        self.params = None
        self.best_model = None
        self.best_score = 0

    def warmup(self, training: Dict[int, WarmStartUser], interview_length=5):
        self.best_model = None
        self.best_score = 0

        if not self.params:
            param_scores = []
            for params in get_combinations({
                'k': [1, 2, 5, 10],
                'reg': [0.001]
            }):
                logger.info(f'Fitting FMF with params {params}')
                self.model = FMF(n_users=self.n_users, n_entities=self.n_entities, max_depth=interview_length,
                                 n_latent_factors=params['k'], regularization=params['reg'])

                best_score = self._fit(training, n_iterations=10)
                param_scores.append((params, best_score))

            best_params, _ = list(sorted(param_scores, key=lambda x: x[1], reverse=True))[0]
            logger.info(f'Found best params for FMF: {best_params}')
            self.model = FMF(n_users=self.n_users, n_entities=self.n_entities, max_depth=interview_length,
                             n_latent_factors=best_params['k'], regularization=best_params['reg'])
            self._fit(training)
        else:
            self.model = FMF(n_users=self.n_users, n_entities=self.n_entities, max_depth=interview_length,
                             n_latent_factors=self.params['k'], regularization=self.params['reg'])
            self._fit(training)

    def _fit(self, users: Dict[int, WarmStartUser], n_iterations=5) -> float:
        self.best_score = 0
        R = get_rating_matrix(users, self.n_users, self.n_entities)
        candidates = self.meta.get_question_candidates(users, limit=self.n_candidates)

        for iteration in tqdm(range(n_iterations), desc=f'[Training FMF]'):
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
