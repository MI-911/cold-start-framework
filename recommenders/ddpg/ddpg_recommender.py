from typing import List, Dict

from loguru import logger

from recommenders.base_recommender import RecommenderBase
from recommenders.ddpg.agent import DDPGAgent
from recommenders.ddpg.utils import OUNoise
from shared.meta import Meta
from shared.user import WarmStartUser

import numpy as np


def get_state(data: Dict, n_entities: int):
    state = np.zeros((n_entities,))
    liked_entities = [e for e, r in data.items() if r == 1]
    left_out = np.random.choice(liked_entities)

    for entity, rating in data.items():
        state[entity] = rating

    return state, left_out


def get_rank(ranking, left_out):
    sorted_ranking = list(sorted(enumerate(ranking), key=lambda x: x[1], reverse=True))
    ranking_dict = {e: i for i, (e, r) in enumerate(sorted_ranking)}
    return ranking_dict[left_out]


def git_hitrate_reward(ranking, left_out, n=10):
    sorted_ranking = list(sorted(enumerate(ranking), key=lambda x: x[1], reverse=True))
    return 1 if left_out in [i for i, r in sorted_ranking[:n]] else 0


class DDPGDirectRankingRecommender(RecommenderBase):
    """
    A DDPG recommender that outputs a direct ranking of all entities.
    The variant to this model is a DDPG recommender that outputs a
    ranking vector that, when multiplied onto some entity embeddings,
    rankings the entities appropriately.
    """
    def __init__(self, meta: Meta):
        super(DDPGDirectRankingRecommender, self).__init__(meta)
        self.n_users = len(meta.users)
        self.n_entities = len(meta.entities)

        self.agent = DDPGAgent(n_entities=self.n_entities, fc1_dims=256, fc2_dims=128)
        self.noise = OUNoise(action_space=self.n_entities)

    def fit(self, training: Dict[int, WarmStartUser]):

        batch_size = 128

        for episode in range(10):
            rewards = []
            ranks = []
            self.noise.reset()

            for step, (user, data) in enumerate(training.items()):
                state, left_out = get_state(data.training, self.n_entities)
                ranking = self.agent.get_action(state)
                ranking = self.noise.get_action(ranking, step)

                # Get the reward
                reward = 1.0 / get_rank(ranking, left_out)
                rank = get_rank(ranking, left_out)

                self.agent.memory.push(state, ranking, reward, state, False)

                if len(self.agent.memory) > batch_size:
                    self.agent.update(batch_size)

                rewards.append(reward)
                ranks.append(rank)

                if step % 50 == 0:
                    logger.info(f'Avg. reward: {np.mean(rewards[len(ranks) - 50:])}')
                    logger.info(f'Avg. rank: {np.mean(ranks[len(ranks) - 50:])}')

    def predict(self, items: List[int], answers: Dict[int, int]) -> Dict[int, float]:
        # Generate state from the answers
        state = np.zeros((self.n_entities,))
        for entity, rating in answers.items():
            state[entity] = rating

        ranking = self.agent.get_action(state)
        ranking = ranking[items]

        return {e: r for e, r in enumerate(ranking)}
