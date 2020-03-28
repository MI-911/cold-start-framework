from enum import Enum
from typing import Dict
import numpy as np
from models.shared.base_recommender import RecommenderBase
from models.shared.meta import Meta
from models.shared.user import WarmStartUser


class Rewards(Enum):
    HIT = 'HIT'
    NDCG = 'NDCG'
    SERENDIPITY = 'SERENDIPITY'
    COVERAGE = 'COVERAGE'


def get_rating_matrix(training, n_users, n_entities, rating_map=None):
    """
    Returns an [n_users x n_entities] ratings matrix.
    """
    if rating_map is None:
        rating_map = {
            1: 1,
            0: 0,
            -1: 0
        }

    R = np.ones((n_users, n_entities)) * rating_map[0]
    for user, data in training.items():
        for entity, rating in data.training.items():
            R[user, entity] = rating_map[rating]

    return R


class Environment:
    """
    Class to simulate the environment a DQN will interact with.
    The environment allows the model to ask questions for a user, receive
    answers, and a reward when the interview is over.
    """
    def __init__(self, recommender: RecommenderBase, reward: Rewards, meta: Meta):
        """
        Constructs an environment object.
        :param recommender: The recommendation model used to generate rewards.
                            When an interview is done, the environment provides the model
                            with the interview answers as ratings, and the environment uses
                            the provided reward metric to generate a reward from the ranking list.
        :param reward: The metric to be used to generate rewards. Whatever metric is provided
                       is the metric the model is learned to optimise.
        """
        # TODO: Allow the environment to handle multiple users in parallel

        self.recommender = recommender
        self.reward = reward
        self.n_users = len(meta.users)
        self.n_entities = len(meta.entities)
        self.ratings = np.zeros((self.n_users, self.n_entities))
        self.state = self.reset()

    def warmup(self, training: Dict[int, WarmStartUser], interview_length: int):
        """
        Trains the recommender model.
        """
        self.ratings = get_rating_matrix(training, self.n_users, self.n_entities)
        self.recommender.warmup(training, interview_length=interview_length)

    def reset(self):
        self.state = np.zeros((self.n_entities * 2,))
        return self.state

    def ask(self, user, entity):
        answer = self.ratings[user, entity]

        # Adjust state
        self.state[entity] = 1
        self.state[entity + 1] = answer

        # Calculate reward
        reward = np.random.rand()  # TODO: Calculate reward from a ranking list
        return self.state, reward

