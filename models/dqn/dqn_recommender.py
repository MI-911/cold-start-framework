from collections import defaultdict
from typing import List, Dict, Union
import numpy as np
from loguru import logger
from tqdm import tqdm

from models.dqn.dqn import DeepQNetwork
from models.dqn.dqn_agent import DqnAgent
from models.dqn.dqn_environment import Environment, Rewards
from models.shared.base_recommender import RecommenderBase
from models.shared.meta import Meta
from models.shared.user import WarmStartUser


def choose_candidates(ratings: Dict[int, WarmStartUser], n=100):
    entity_ratings = {}

    for user, data in ratings.items():
        for entity, rating in data.training.items():
            if entity not in entity_ratings:
                entity_ratings[entity] = 0
            entity_ratings[entity] += 1

    sorted_entity_ratings = sorted(entity_ratings.items(), key=lambda x: x[1], reverse=True)
    return [e for e, r in sorted_entity_ratings][:n]


class DqnRecommender(RecommenderBase):
    """
    A recommendation interviewer that uses reinforcement learning to learn
    how to interview users. Utilises an underlying recommendation model
    to generate rewards.

    Note that the models used with a DQN recommender must be user-agnostic.
    The models should be able to generate recommendations only given the raw
    ratings of a new user, and nothing more. Latent models like MF will have to
    learn a new embedding for every such user.
    """
    def __init__(self, meta: Meta, recommender: RecommenderBase):
        super(DqnRecommender, self).__init__(meta)

        # Allocate environment
        self.recommender = recommender
        self.environment: Union[RecommenderBase, None] = None

        # Allocate DQN agent
        self.agent: Union[DqnAgent, None] = None

        self.n_users = len(self.meta.users)
        self.n_entities = len(self.meta.entities)

    def warmup(self, training: Dict[int, WarmStartUser], interview_length=5) -> None:
        # Construct environment
        self.environment = Environment(
            recommender=self.recommender, reward=Rewards.NDCG, meta=self.meta)

        # Construct model
        self.agent = DqnAgent(
            candidates=choose_candidates(training),
            n_entities=self.n_users,
            batch_size=64,
            alpha=0.0003,
            gamma=1.0,
            epsilon=1.0,
            eps_end=0.1,
            eps_dec=0.996
        )

        # Train the recommender model
        self.environment.warmup(training, interview_length)

        # Train the DQN
        self.fit_dqn(training, interview_length)

    def fit_dqn(self, training: Dict[int, WarmStartUser], interview_length: int) -> None:
        n_iterations = 10

        scores = []
        epsilons = []

        for i in range(n_iterations):
            logger.info(f'DQN starting iteration {i}...')
            users = list(training.keys())
            np.random.shuffle(users)

            for user in tqdm(users, desc=f'Training on users (avg. score: {np.mean(scores[len(scores) - 10])})'):
                epsilons.append(self.agent.epsilon)

                score = 0
                state = self.environment.reset()

                for q in range(interview_length):
                    question = self.agent.choose_action(state)
                    new_state, reward = self.environment.ask(user, question)

                    # Accumulate the rewards
                    score += reward
                    scores.append(score)

                    # Store the memory for training
                    self.agent.store_memory(state, question, new_state, reward, q == interview_length)
                    state = new_state

                # Train on the memories
                self.agent.learn()

    def interview(self, answers: Dict, max_n_questions=5) -> List[int]:
        pass

    def predict(self, items: List[int], answers: Dict) -> Dict[int, float]:
        pass

    def get_parameters(self):
        pass

    def load_parameters(self, params):
        pass


