import pickle

from torch import nn
import torch as tt
import torch.nn.functional as ff

from models.ddpg.memory import ReplayMemory
from models.ddpg.model import Actor, Critic
import torch as tt
import torch.optim as optim

from models.ddpg.noise import OrnsteinUhlenbeckProcess
from models.ddpg.utils import to_numpy, to_tensor, soft_update
import numpy as np


class DDPGAgent:
    def __init__(self, embedding_size, state_size):
        self.action_size = embedding_size
        self.state_size = state_size

        self.actor = Actor(action_size=self.action_size, state_size=state_size, fc1=512, fc2=256)
        self.actor_target = pickle.loads(pickle.dumps(self.actor))
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.001)

        self.critic = Critic(state_size=state_size, action_size=self.action_size, fc1=512, fc2=256)
        self.critic_target = pickle.loads(pickle.dumps(self.critic))
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.001)
        self.critic_criterion = ff.smooth_l1_loss

        self.memory = ReplayMemory(max_mem_size=100000, action_size=self.action_size, state_size=self.state_size)
        self.N = OrnsteinUhlenbeckProcess(size=self.action_size, theta=0.15, mu=0.0, sigma=0.2)  # Noise generator

        self.batch_size = 128
        self.tau = 0.001
        self.discount = 1.0

        self.epsilon = 1.0
        self.eps_dec = .996
        self.eps_end = 0.1

    def choose_action(self, state, decay_epsilon=True, add_noise=True):
        state = to_tensor(state)
        action = to_numpy(self.actor(state))

        if add_noise:
            action += self.N.sample()

        if decay_epsilon:
            self.epsilon = min(self.epsilon * self.eps_dec, self.eps_end)

        action = np.clip(action, -1., 1.)
        return action

    def random_action(self):
        action = np.random.uniform(-1., 1., self.action_size)
        return action

    def update_policy(self):
        # Sample a batch of memories
        states, actions, new_states, rewards, terminals = self.memory.batch(256)

        # Calculate the Q values for the next states
        next_q_values = self.critic_target(
            [to_tensor(new_states),
            self.actor_target(to_tensor(new_states))]
        )

        # Calculate the target Q values that the critic and actor should try to reach
        rewards = to_tensor(rewards).view(-1, 1)
        target_q_values = rewards + self.discount * to_tensor(terminals.astype(np.float32)).view(-1, 1) * next_q_values

        # Update the critic network
        self.critic.zero_grad()
        q_values = self.critic([to_tensor(states), to_tensor(actions)])
        critic_loss = self.critic_criterion(q_values, target_q_values)
        critic_loss.backward()
        tt.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # Update the actor network
        self.actor.zero_grad()
        actor_loss = -self.critic([to_tensor(states), self.actor(to_tensor(states))])
        actor_loss = actor_loss.mean()
        actor_loss.backward()
        tt.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        # Update targets
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        # Return losses
        return to_numpy(critic_loss), to_numpy(actor_loss)

    def eval(self):
        self.critic.eval()
        self.critic_target.eval()
        self.actor.eval()
        self.actor_target.eval()

    def train(self):
        self.critic.train()
        self.critic_target.train()
        self.actor.train()
        self.actor_target.train()
