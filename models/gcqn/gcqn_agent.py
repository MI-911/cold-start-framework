import torch as tt
import numpy as np
from typing import List

from dgl import DGLGraph, DGLHeteroGraph
from loguru import logger

from models.dqn.dqn import DeepQNetwork
import pickle

from models.gcqn.gcqn import HeteroRGCN


class GcqnAgent:
    def __init__(self, G: DGLHeteroGraph, memory_size, interview_length, action_size, batch_size, n_entities, candidates):
        self.G = G
        self.candidates = candidates

        # Allocate replay memory stores
        self.memory_size = memory_size

        self.liked_memory = np.zeros((self.memory_size, interview_length))
        self.unknown_memory = np.zeros((self.memory_size, interview_length))
        self.disliked_memory = np.zeros((self.memory_size, interview_length))

        self.action_memory = np.zeros((self.memory_size, action_size), dtype=np.uint8)

        self.new_liked_memory = np.zeros((self.memory_size, interview_length))
        self.new_unknown_memory = np.zeros((self.memory_size, interview_length))
        self.new_disliked_memory = np.zeros((self.memory_size, interview_length))

        self.reward_memory = np.zeros((self.memory_size,), dtype=np.float32)
        self.terminal_memory = np.zeros((self.memory_size,), dtype=np.uint8)

        self.mem_counter = 0
        self.batch_size = batch_size
        self.action_size = action_size
        self.action_space = np.arange(self.action_size)

        self.epsilon = 1.0
        self.eps_dec = .996
        self.eps_end = 0.2

        self.gamma = 1.0

        self.Q_eval = HeteroRGCN(self.G, in_size=10, hidden_size=10, embedding_size=10)
        self.Q_target = pickle.loads(pickle.dumps(self.Q_eval))

    def store_memory(self, liked, unknown, disliked, action, new_liked, new_unknown, new_disliked, reward, terminal):
        # Override old memories when we reach max_mem_size
        i = self.mem_counter % self.memory_size

        # Construct action vector
        actions = np.zeros((self.action_size,), dtype=np.uint8)
        actions[action] = 1

        # Store the memory
        self.liked_memory[i] = liked
        self.unknown_memory[i] = unknown
        self.disliked_memory[i] = disliked

        self.action_memory[i] = actions

        self.new_liked_memory[i] = new_liked
        self.new_unknown_memory[i] = new_unknown
        self.new_disliked_memory[i] = new_disliked

        self.reward_memory[i] = reward
        self.terminal_memory[i] = terminal

        # Increment the counter
        self.mem_counter += 1

    def choose_action(self, liked, unknown, disliked, explore=True) -> int:
        # Make the DQN predict action rewards given this state

        with tt.no_grad():
            predicted_rewards = self.Q_eval(self.G, liked, unknown, disliked)

            if np.random.rand() > self.epsilon or not explore:
                # Exploit
                return int(tt.argmax(predicted_rewards[-1][self.candidates]))
            else:
                # Explore
                return np.random.choice(self.candidates)

    def learn(self):
        if self.mem_counter < self.batch_size:
            # Continue until we have enough memories for a full batch
            return

        # Generate a random batch of memory indices (where all indices < self.mem_counter)
        max_memory_index = self.mem_counter if self.mem_counter < self.memory_size else self.memory_size
        batch_indices = np.random.choice(max_memory_index, self.batch_size)

        l_batch = self.liked_memory[batch_indices]
        u_batch = self.unknown_memory[batch_indices]
        d_batch = self.disliked_memory[batch_indices]

        action_batch = self.action_memory[batch_indices]

        n_l_batch = self.new_liked_memory[batch_indices]
        n_u_batch = self.new_unknown_memory[batch_indices]
        n_d_batch = self.new_disliked_memory[batch_indices]

        reward_batch = self.reward_memory[batch_indices]
        terminal_batch = self.terminal_memory[batch_indices]

        # Convert action memories to indices so we use them to index directly later
        action_values = np.arange(self.action_size, dtype=np.uint8)
        action_indices = np.dot(action_batch, action_values)

        # Predict rewards for the current state and the next one
        current_predicted_rewards = self.Q_eval(self.G, l_batch, u_batch, d_batch)
        target_rewards = current_predicted_rewards.clone().cpu().detach().numpy()
        next_predicted_rewards = self.Q_target(self.G, n_l_batch, n_u_batch, n_d_batch)  # Should come from the target network

        # Index trickery so we can index the in-batch tensors
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        # Construct the optimal predicted rewards (the "ground truth" labels for every memory)
        target_update = reward_batch + self.gamma * tt.max(next_predicted_rewards, dim=1)[0].cpu().detach().numpy() * terminal_batch
        for i in range(len(batch_index)):
            target_rewards[batch_index[i], action_indices[i]] = target_update[i]

        # Adjust the model
        self.Q_eval.zero_grad()
        # self.Q_target.zero_grad()

        loss = self.Q_eval.get_loss(current_predicted_rewards, tt.tensor(target_rewards))
        loss.backward()
        # tt.nn.utils.clip_grad_norm_(self.Q_eval.parameters(), 1.0)
        self.Q_eval.optimizer.step()

        # Decrease the chance that we will explore random questions in the future
        self.epsilon = self.epsilon * self.eps_dec if self.epsilon > self.eps_end else self.eps_end

        # Do a soft update
        tau = 1e-3
        for target_param, eval_param in zip(self.Q_target.parameters(), self.Q_eval.parameters()):
            target_param.data.copy_(tau * eval_param.data + (1.0 - tau) * target_param.data)

        # Check if we should update the target network
        # if self.mem_counter % 500 == 0:
        #     self.Q_target.load_state_dict(self.Q_eval.state_dict())

        return loss
