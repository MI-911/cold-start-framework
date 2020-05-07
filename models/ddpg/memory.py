import numpy as np

from models.ddpg.utils import to_tensor


class ReplayMemory:
    def __init__(self, max_mem_size, state_size, action_size):
        self.state_memory = np.zeros((max_mem_size, state_size), dtype=np.float32)
        self.action_memory = np.zeros((max_mem_size, action_size), dtype=np.float32)
        self.new_state_memory = np.zeros((max_mem_size, state_size), dtype=np.float32)
        self.reward_memory = np.zeros((max_mem_size,), dtype=np.float32)
        self.terminal_memory = np.zeros((max_mem_size,), dtype=np.uint8)

        self.mem_counter = 0
        self.max_mem_size = max_mem_size

    def store(self, state, action, new_state, reward, terminal):
        i = self.mem_counter % self.max_mem_size
        self.state_memory[i] = state
        self.action_memory[i] = action
        self.new_state_memory[i] = new_state
        self.reward_memory[i] = reward
        self.terminal_memory[i] = terminal

        self.mem_counter += 1

    def batch(self, size):
        max_memory_index = self.mem_counter if self.mem_counter < self.max_mem_size else self.max_mem_size
        batch_indices = np.random.choice(max_memory_index, size)

        return (mem[batch_indices] for mem in
                [self.state_memory, self.action_memory,
                 self.new_state_memory, self.reward_memory,
                 self.terminal_memory])
