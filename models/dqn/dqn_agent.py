import torch as tt
import numpy as np


class DqnAgent:
    def __init__(self, gamma, epsilon, alpha, n_entities, batch_size, max_mem_size=100000, eps_end=0.01, eps_dec=0.996):
        """
        Constructs an agent for a DQN learning process
        @param gamma: The discount rates for future predicted rewards.
                      If 1, future rewards are just as important as current ones.
        @param epsilon: The initial probability that the agent will explore a random question.
                        Epsilon decreases over time by eps_dec at every batch iteration.
        @param alpha: The learning rate.
        @param n_entities: The number of entities in the system.
        @param batch_size: The number of transitions in a batch during learning.
        @param max_mem_size: The maximum number of stored transitions. If the size of stored
                             transitions exceeds this value, they are replaced by new ones.
        @param eps_end: The minimum epsilon value.
        @param eps_dec: The decrease factor for epsilon. At every iteration, epsilon is decreased to epsilon * eps_dec.
        """
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_end = eps_end
        self.eps_dec = eps_dec
        self.alpha = alpha
        self.n_entities = n_entities
        self.batch_size = batch_size
        self.max_mem_size = max_mem_size

        self

