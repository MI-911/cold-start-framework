from collections import defaultdict
from typing import List, Dict, Union
import numpy as np
from loguru import logger
from tqdm import tqdm

from models.base_interviewer import InterviewerBase
from models.dqn.dqn import DeepQNetwork
from models.dqn.dqn_agent import DqnAgent
from models.dqn.dqn_environment import Environment, Rewards
from recommenders.base_recommender import RecommenderBase
from shared.meta import Meta
from shared.user import WarmStartUser


def choose_candidates(ratings: Dict[int, WarmStartUser], n=100):
    entity_ratings = {}

    for user, data in ratings.items():
        for entity, rating in data.training.items():
            if entity not in entity_ratings:
                entity_ratings[entity] = 0
            entity_ratings[entity] += 1

    sorted_entity_ratings = sorted(entity_ratings.items(), key=lambda x: x[1], reverse=True)
    return [e for e, r in sorted_entity_ratings][:n]


class DqnInterviewer(InterviewerBase):
    """
    A recommendation interviewer that uses reinforcement learning to learn
    how to interview users. Utilises an underlying recommendation model
    to generate rewards.

    Note that the models used with a DQN recommender must be user-agnostic.
    The models should be able to generate recommendations only given the raw
    ratings of a new user, and nothing more. Latent models like MF will have to
    learn a new embedding for every such user.
    """

    def __init__(self, meta: Meta, recommender, use_cuda: bool):
        super(DqnInterviewer, self).__init__(meta)

        self.use_cuda = use_cuda

        # Allocate environment
        if not recommender:
            raise RuntimeError('No underlying recommender provided to the naive interviewer.')

        self.recommender: RecommenderBase = recommender(meta)
        self.environment: Union[RecommenderBase, None] = None

        # Allocate DQN agent
        self.agent: Union[DqnAgent, None] = None

        self.n_users = len(self.meta.users)
        self.n_entities = len(self.meta.entities)

    def warmup(self, training: Dict[int, WarmStartUser], interview_length=5) -> None:
        # Construct environment
        self.environment = Environment(
            recommender=self.recommender, reward_metric=Rewards.NDCG, meta=self.meta)

        # Construct model
        self.agent = DqnAgent(
            candidates=choose_candidates(training),
            n_entities=self.n_entities,
            batch_size=64,
            alpha=0.0003,
            gamma=1.0,
            epsilon=1.0,
            eps_end=0.1,
            eps_dec=0.996,

        )

        # Train the recommender model
        self.environment.warmup(training)

        # Train the DQN
        self.fit_dqn(training, interview_length)

    def fit_dqn(self, training: Dict[int, WarmStartUser], interview_length: int) -> None:
        n_iterations = 10

        epsilons = []
        scores = []
        losses = []

        def recent_mean(lst):
            if l := len(lst) == 0:
                return 'NaN'
            recent = lst[l - 10:] if l >= 10 else lst
            return np.mean(recent)

        for i in range(n_iterations):
            logger.info(f'DQN starting iteration {i}...')

            users = list(training.keys())
            np.random.shuffle(users)

            t = tqdm(users)
            for user in t:

                t.set_description(f'Training on users (Scores: {recent_mean(scores)}, Loss: {recent_mean(losses)}, '
                                  f'Epsilon: {recent_mean(epsilons)})')

                state = self.environment.reset()
                self.environment.select_user(user)

                # Train on the memories
                _loss = self.agent.learn()
                _reward = 0

                for q in range(interview_length):
                    question = self.agent.choose_action(state)
                    new_state, reward = self.environment.ask(user, question)

                    # Store the memory for training
                    self.agent.store_memory(state, question, new_state, reward, q == interview_length)
                    state = new_state

                    _reward += reward

                # Record scores, losses, rewards
                epsilons.append(self.agent.epsilon)
                scores.append(_reward)
                if _loss is not None:
                    losses.append(_loss.detach().numpy())

    def interview(self, answers: Dict, max_n_questions=5) -> List[int]:
        state = np.zeros((self.n_entities * 2,), dtype=np.float32)
        for entity, rating in answers.items():
            entity_state_idx = entity * 2
            state[entity_state_idx] = 1
            state[entity_state_idx + 1] = rating

        question = self.agent.choose_action(state)
        return [question]

    def predict(self, items: List[int], answers: Dict) -> Dict[int, float]:
        return self.environment.recommender.predict(items, answers)

    def get_parameters(self):
        return {}

    def load_parameters(self, params):
        pass
