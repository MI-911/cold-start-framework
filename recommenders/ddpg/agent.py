import pickle

import torch as tt
import numpy as np

from recommenders.ddpg.actor import Actor
from recommenders.ddpg.critic import Critic
from recommenders.ddpg.utils import Memory


class DDPGAgent:
    def __init__(self, n_entities, fc1_dims, fc2_dims, actor_lr=1e-4, critic_lr=1e-3, gamma=0.99, tau=1e-2,
                 max_memory_size=1000):
        self.n_entities = n_entities
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.tau = tau
        self.max_memory_size = max_memory_size

        # Initialise actor and critic networks
        self.actor = Actor(state_dims=self.n_entities, fc1_dims=1024, fc2_dims=512, action_dims=self.n_entities)
        self.critic = Critic(state_dims=self.n_entities, action_dims=self.n_entities, fc1_dims=1024, fc2_dims=512,
                             output_dims=self.n_entities)

        # Create targets to stabilize training a bit
        self.actor_target = pickle.loads(pickle.dumps(self.actor))
        self.critic_target = pickle.loads(pickle.dumps(self.critic))

        # Replay memory and optimisers
        self.memory = Memory(self.max_memory_size)
        self.critic_criterion = tt.nn.MSELoss()
        self.actor_optimizer = tt.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = tt.optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        self.n_updates = 0

    def get_action(self, state):
        state = tt.from_numpy(state).float()
        action = self.actor(state)
        return action.cpu().detach().numpy()

    def update(self, batch_size):
        states, actions, rewards, next_states, _ = self.memory.sample(batch_size)
        states = tt.tensor(states, dtype=tt.float32)
        actions = tt.tensor(actions, dtype=tt.float32)
        rewards = tt.tensor(rewards, dtype=tt.float32)
        next_states = tt.tensor(next_states, dtype=tt.float32)

        # Get loss for the critic network
        Q_values = self.critic(states, actions)
        next_actions = self.actor_target(next_states)
        Q_next = self.critic_target(next_states, next_actions)
        Q_prime = rewards + self.gamma * Q_next.cpu()
        critic_loss = self.critic_criterion(Q_values, Q_prime.to(self.critic.device))

        # Get loss for the actor network
        actor_loss = -self.critic(states, self.actor(states)).mean()

        # Update the networks
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update the target networks as well (consider delaying this a bit)
        if self.n_updates % 20 == 0:
            self.actor_target = pickle.loads(pickle.dumps(self.actor))
            self.critic_target = pickle.loads(pickle.dumps(self.critic))

        self.n_updates += 1
