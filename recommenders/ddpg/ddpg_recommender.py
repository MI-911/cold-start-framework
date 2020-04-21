from typing import List, Dict

from recommenders.base_recommender import RecommenderBase
from recommenders.ddpg.actor import Actor
from recommenders.ddpg.critic import Critic
from shared.meta import Meta
from shared.user import WarmStartUser

import pickle


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

        self.actor = self.actor_target = None
        self.critic = self.critic_target = None

    def fit(self, training: Dict[int, WarmStartUser]):
        # Initialise actor and critic networks
        self.actor = Actor(state_dims=self.n_entities, fc1_dims=256, fc2_dims=128, action_dims=self.n_entities)
        self.critic = Critic(state_dims=self.n_entities, action_dims=self.n_entities, fc1_dims=256, fc2_dims=128,
                             output_dims=self.n_entities)
        self.actor_target = pickle.loads(pickle.dumps(self.actor))
        self.critic_target = pickle.loads(pickle.dumps(self.critic))

    def predict(self, items: List[int], answers: Dict[int, int]) -> Dict[int, float]:
        # Rank the items given these answers
        pass