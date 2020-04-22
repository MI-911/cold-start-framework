import torch as tt
import numpy as np
from typing import List

from models.dqn.dqn import DeepQNetwork
import pickle


class DqnAgent:
    def __init__(self, gamma: float, epsilon: float, alpha: float, candidates: List[int], n_entities: int,
                 batch_size: int, fc1_dims: int,  max_mem_size: float = 1000, eps_end: float = 0.01, eps_dec: float = 0.996,
                 use_cuda: bool = False):
        """
        Constructs an agent for a DQN learning process
        :param gamma: The discount rates for future predicted rewards.
                      If 1, future rewards are just as important as current ones.
        :param epsilon: The initial probability that the agent will explore a random question.
                        Epsilon decreases over time by eps_dec at every batch iteration.
        :param alpha: The learning rate.
        :param n_entities: The number of entities.
        :param candidates: The question candidates.
        :param batch_size: The number of transitions in a batch during learning.
        :param max_mem_size: The maximum number of stored transitions. If the size of stored
                             transitions exceeds this value, they are replaced by new ones.
        :param eps_end: The minimum epsilon value.
        :param eps_dec: The decrease factor for epsilon. At every iteration, epsilon is decreased to epsilon * eps_dec.
        """
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_end = eps_end
        self.eps_dec = eps_dec
        self.alpha = alpha
        self.candidates = candidates
        self.n_entities = n_entities
        self.batch_size = batch_size
        self.max_mem_size = max_mem_size

        self.action_size = n_entities  # One action (question) for every entity
        self.state_size = self.action_size * 2  # One question and one answer for every entity

        # Allocate DQN model
        self.Q_eval = DeepQNetwork(
            state_size=self.state_size, fc1_dims=fc1_dims, fc2_dims=fc1_dims // 2, actions_size=self.action_size,
            alpha=alpha, use_cuda=use_cuda)

        # Allocate a target network to stabilise training
        self.Q_target = self._synchronize()

        # Allocate memory containers
        self.state_memory = np.zeros((self.max_mem_size, self.state_size), dtype=np.float32)
        self.action_memory = np.zeros((self.max_mem_size, self.action_size), dtype=np.uint8)
        self.new_state_memory = np.zeros((self.max_mem_size, self.state_size), dtype=np.float32)
        self.reward_memory = np.zeros((self.max_mem_size,), dtype=np.float32)
        self.terminal_memory = np.zeros((self.max_mem_size,), dtype=np.uint8)

        self.mem_counter = 0

    def _synchronize(self):
        return pickle.loads(pickle.dumps(self.Q_eval))

    def store_memory(self, state: np.ndarray, action: int, new_state: np.ndarray, reward: float, terminal: bool):
        # Override old memories when we reach max_mem_size
        i = self.mem_counter % self.max_mem_size

        # Construct action vector
        actions = np.zeros((self.action_size,), dtype=np.uint8)
        actions[action] = 1

        # If the memory is already occupied, explicitly delete it to GC immediately
        if self.state_memory[i] is not None:
            o_state = self.state_memory[i]
            o_action = self.action_memory[i]
            o_new_state = self.new_state_memory[i]
            o_reward = self.reward_memory[i]
            o_terminal = self.terminal_memory[i]

            del o_state
            del o_action
            del o_new_state
            del o_reward
            del o_terminal

        # Store the memory
        self.state_memory[i] = state
        self.action_memory[i] = actions
        self.new_state_memory[i] = new_state
        self.reward_memory[i] = reward
        self.terminal_memory[i] = terminal

        # Increment the counter
        self.mem_counter += 1

    def choose_action(self, state: np.ndarray) -> int:
        # Make the DQN predict action rewards given this state
        predicted_rewards = self.Q_eval(state.reshape((1, state.shape[0])))

        if np.random.rand() > self.epsilon:
            # Exploit
            return int(tt.argmax(predicted_rewards))
        else:
            # Explore
            return np.random.choice(self.candidates)

    def learn(self):
        if self.mem_counter < self.batch_size:
            # Continue until we have enough memories for a full batch
            return

        # Generate a random batch of memory indices (where all indices < self.mem_counter)
        max_memory_index = self.mem_counter if self.mem_counter < self.max_mem_size else self.max_mem_size
        batch_indices = np.random.choice(max_memory_index, self.batch_size)

        state_batch = self.state_memory[batch_indices]
        action_batch = self.action_memory[batch_indices]
        new_state_batch = self.new_state_memory[batch_indices]
        reward_batch = self.reward_memory[batch_indices]
        terminal_batch = self.terminal_memory[batch_indices]

        # Convert action memories to indices so we use them to index directly later
        action_values = np.arange(self.n_entities, dtype=np.uint8)
        action_indices = np.dot(action_batch, action_values)

        # Predict rewards for the current state and the next one
        current_predicted_rewards = self.Q_eval(state_batch)
        target_rewards = current_predicted_rewards.clone().cpu().detach().numpy()
        next_predicted_rewards = self.Q_target(new_state_batch)  # Should come from the target network

        # Index trickery so we can index the in-batch tensors
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        # Construct the optimal predicted rewards (the "ground truth" labels for every memory)
        target_update = reward_batch + self.gamma * tt.max(next_predicted_rewards, dim=1)[0].cpu().detach().numpy() * terminal_batch
        for i in range(len(batch_index)):
            target_rewards[batch_index[i], action_indices[i]] = target_update[i]

        # Adjust the model
        loss = self.Q_eval.get_loss(current_predicted_rewards, tt.tensor(target_rewards))
        loss.backward()
        self.Q_eval.optimizer.step()
        self.Q_eval.zero_grad()

        # Decrease the chance that we will explore random questions in the future
        self.epsilon = self.epsilon * self.eps_dec if self.epsilon > self.eps_end else self.eps_end

        # Check if we should update the target network
        if self.mem_counter % 100 == 0:
            self.Q_target = self._synchronize()

        return loss
