import torch as tt
import torch.nn as nn
import torch.nn.functional as ff


class Actor(nn.Module):
    """
    The actor in a Deep Deterministic Policy Gradient setting.
    The actor takes a state and outputs an action that may be
    continuous and highly dimensional.
    """
    def __init__(self, state_dims: int, fc1_dims: int, fc2_dims: int, action_dims: int):
        super(Actor, self).__init__()
        self.state_input = nn.Linear(state_dims, fc1_dims)
        self.hidden = nn.Linear(fc1_dims, fc2_dims)
        self.action_output = nn.Linear(fc2_dims, action_dims)

    def forward(self, state):
        """
        Generates an action vector from a state
        """
        x = ff.relu(self.state_input(state))
        x = ff.relu(self.hidden(x))
        return tt.tanh(self.action_output(x))
