import torch as tt
import torch.nn as nn
import torch.nn.functional as ff


class Critic(nn.Module):
    def __init__(self, state_dims: int, action_dims: int, fc1_dims: int, fc2_dims: int, output_dims: int):
        super(Critic, self).__init__()
        self.state_action_input = nn.Linear(state_dims + action_dims, fc1_dims)
        self.hidden = nn.Linear(fc1_dims, fc2_dims)
        self.output = nn.Linear(fc2_dims, output_dims)

    def forward(self, state, action):
        """
        Criticises the act of taking this action in this
        state by returning a reward.
        """
        x = tt.cat([state, action], 1)
        x = ff.relu(self.state_action_input(x))
        x = ff.relu(self.hidden(x))
        return self.output(x)