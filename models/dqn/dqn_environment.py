from enum import Enum
from typing import Dict
import numpy as np

from recommenders.base_recommender import RecommenderBase
from shared.meta import Meta
from shared.user import WarmStartUser

from experiments.metrics import ndcg_at_k, ser_at_k


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

    R = np.zeros((n_users, n_entities))
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
    def __init__(self, recommender: RecommenderBase, reward_metric: Rewards, meta: Meta):
        """
        Constructs an environment object.
        :param recommender: The recommendation model used to generate rewards.
                            When an interview is done, the environment provides the model
                            with the interview answers as ratings, and the environment uses
                            the provided reward metric to generate a reward from the ranking list.
        :param reward_metric: The metric to be used to generate rewards. Whatever metric is provided
                              is the metric the model is learned to optimise.
        """
        # TODO: Allow the environment to handle multiple users in parallel

        self.meta = meta
        self.recommender = recommender
        self.reward_metric = reward_metric
        self.n_users = len(meta.users)
        self.n_entities = len(meta.entities)
        self.ratings = np.zeros((self.n_users, self.n_entities))
        self.top_pop_movies = None

        self.state = []    # State keeping for the DQN agent (a numpy array)
        self.answers = {}  # State keeping for the environment (fits to the recommender.predict() method)

        self.current_user = None   # The user currently being interviewed.
        self.left_out_item = None  # The left out item. When resetting, we leave out a random liked entity for the user.
        self.to_rate = None        # The 101 items to rate

        self.reset()

    def warmup(self, training: Dict[int, WarmStartUser]):
        """
        Trains the recommender model.
        """
        self.ratings = get_rating_matrix(training, self.n_users, self.n_entities)
        self.top_pop_movies = self._top_pop_movies()
        self.recommender.fit(training=training)

    def reset(self):
        # Nullify the current user, re-insert their rating into the ratings matrix
        if self.current_user and self.left_out_item:
            self.ratings[self.current_user, self.left_out_item] = 1
        self.current_user = None
        self.left_out_item = None
        self.to_rate = None

        # Reset states, return to the agent
        self.state = np.zeros((self.n_entities * 2,), dtype=np.float32)
        self.answers = {}
        return self.state

    def select_user(self, user):
        self.current_user = user

        # Choose a random liked entity as the left-out entity
        positive_samples, = np.where(self.ratings[self.current_user] == 1)
        self.left_out_item = np.random.choice(positive_samples)

        # Select negative samples
        negative_samples, = np.where(self.ratings[self.current_user] == 0)
        self.to_rate = np.random.choice(negative_samples, 100).tolist() + [self.left_out_item]

        # Erase that rating from the ratings matrix so they can't answer it during interviews
        self.ratings[self.current_user, self.left_out_item] = 0

    def ask(self, user, entity):
        answer = self.ratings[user, entity]

        # Adjust state
        entity_state_idx = entity * 2
        self.state[entity_state_idx] = 1
        self.state[entity_state_idx + 1] = answer

        self.answers[entity] = answer

        # Calculate reward
        reward = self._reward()
        return self.state, reward

    def _reward(self):
        scores = self.recommender.predict(self.to_rate, self.answers)
        ranked = [e for e, s in sorted(scores.items(), key=lambda x: x[1], reverse=True)]

        # What should we leave out?
        # 1. Pick a random liked entity that has not yet been answered on
        #   - Issue: There might not be any entities left.
        # 2. When resetting, ask to reset to a specific user.
        #    When this happens, pick, at random, a liked entity and remove
        #    its rating from the rating matrix.
        #    This is the LOO entity that we will generate rewards from.
        #    Also, when resetting, re-insert the old LOO entity into the
        #    ratings matrix.

        metric_score = self._compute_metric(ranked, self.left_out_item)
        return metric_score

    def _compute_metric(self, rankings, left_out):
        k = 10
        relevance = [1 if r == left_out else 0 for r in rankings]

        metric_handlers = {
            Rewards.NDCG: ndcg_at_k(relevance, k=k),
            Rewards.HIT: sum(relevance[:k]),
            Rewards.SERENDIPITY: ser_at_k(zip(rankings, relevance), self.top_pop_movies, k=k)
        }

        return metric_handlers[self.reward_metric]

    def _top_pop_movies(self):
        n_movie_ratings = self.ratings[:, self.meta.recommendable_entities].sum(axis=0)
        sorted_popular_movies = sorted(zip(self.meta.recommendable_entities, n_movie_ratings),
                                       key=lambda x: x[1], reverse=True)

        return [m for m, rs in list(sorted_popular_movies)[:100]]


