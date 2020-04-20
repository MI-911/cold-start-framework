from typing import Dict, List

from tqdm import tqdm

from models.base_interviewer import InterviewerBase
from recommenders.base_recommender import RecommenderBase
from recommenders.mf.mf import MatrixFactorisation
import numpy as np
import random
from loguru import logger

from shared.meta import Meta
from shared.user import WarmStartUser
from shared.utility import get_combinations


def flatten_rating_triples(training: Dict[int, WarmStartUser]):
    training_triples = []
    for u_idx, user in training.items():
        for (entity, rating) in user.training.items():
            training_triples.append((u_idx, entity, rating))

    return training_triples


def convert_rating(rating):
    if rating == 1:
        return 1
    elif rating == -1:
        return 0
    elif rating == 0:
        # We can make a choice here - either return 0 or 0.5
        return 0.5


class MatrixFactorizationRecommender(RecommenderBase):
    def __init__(self, meta: Meta, use_cuda=False):
        super(MatrixFactorizationRecommender, self).__init__(meta)
        self.meta = meta
        self.optimal_params = None
        self.model = None

    def fit(self, training: Dict[int, WarmStartUser]) -> None:
        n_users = len(self.meta.users)
        n_entities = len(self.meta.entities)

        if self.optimal_params is None:
            scores = []
            parameters = {'k': [1, 2, 5, 10]}

            # Find optimal parameters
            for params in get_combinations(parameters):
                logger.info(f'Grid searching MF with params: {params}')
                self.model = MatrixFactorisation(n_users, n_entities, params['k'])
                self._train(training, max_iterations=100)
                score = self._validate(training)
                scores.append((params, score))

            # Use optimal parameters to train a new model
            optimal_params, _ = list(sorted(scores, key=lambda x: x[1], reverse=True))[0]
            self.optimal_params = optimal_params

            logger.info(f'Found best params for MF: {self.optimal_params}')
            self.model = MatrixFactorisation(n_users, n_entities, self.optimal_params['k'])
            self._train(training, max_iterations=100)

        else:
            # Reuse optimal parameters
            logger.info(f'Reusing best params for MF: {self.optimal_params}')
            self.model = MatrixFactorisation(n_users, n_entities, self.optimal_params['k'])
            self._train(training, max_iterations=100)

    def _train(self, users: Dict[int, WarmStartUser], max_iterations=100):
        # Train the model on training samples
        training_triples = flatten_rating_triples(users)
        for iteration in tqdm(range(max_iterations), desc=f'[Training MF]'):
            random.shuffle(training_triples)
            self.model.train_als(training_triples)

    def _validate(self, users: Dict[int, WarmStartUser]):
        # Validate on users using the meta.validator
        predictions = []
        for u_idx, user in users.items():
            prediction = self.model.predict(u_idx, user.validation.to_list())
            predictions.append((user.validation, prediction))

        return self.meta.validator.score(predictions, self.meta)

    def predict(self, items, answers):
        # Predict a user as the avg embedding of the items they liked
        u_embedding_items = [e for e, r in answers.items() if r == 1]
        return self.model.predict_avg_items(u_embedding_items, items)

    def get_parameters(self):
        return self.optimal_params

    def load_parameters(self, params):
        self.optimal_params = params

