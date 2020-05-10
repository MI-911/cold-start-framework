import torch as tt
import torch.nn as nn
import torch.nn.functional as ff

from models.ddpg.utils import to_tensor


class Actor(nn.Module):
    def __init__(self, state_size, fc1, fc2, action_size):
        super(Actor, self).__init__()

        self.input = nn.Linear(state_size, fc1)
        self.fc1 = nn.Linear(fc1, fc2)
        self.output = nn.Linear(fc2, action_size)

        self.relu = nn.ReLU()

        self.device = tt.device('cuda' if tt.cuda.is_available() else 'cpu')
        self.to(self.device)

    def drop_relu(self, input):
        return ff.relu(input)
        # return nn.Dropout(0.5)(ff.relu(input))

    def forward(self, state):
        x = self.drop_relu(self.input(state))
        x = self.drop_relu(self.fc1(x))
        x = tt.tanh(self.output(x))  # Apply an activation function here? Requires normalization of embeddings

        return tt.tanh(x)


class Critic(nn.Module):
    def __init__(self, state_size, action_size, fc1, fc2):
        super(Critic, self).__init__()

        self.input = nn.Linear(state_size, fc1)
        self.fc1 = nn.Linear(fc1 + action_size, fc2)
        self.output = nn.Linear(fc2, 1)
        self.device = tt.device('cuda' if tt.cuda.is_available() else 'cpu')
        self.to(self.device)

    def drop_relu(self, input):
        return ff.relu(input)
        # return nn.Dropout(0.5)(ff.relu(input))

    def forward(self, sa):
        state, action = sa
        # First, process the state alone
        s = self.drop_relu(self.input(state))
        x = self.drop_relu(self.fc1(tt.cat([s, action], dim=1)))
        x = self.output(x)  # Again, activation function here? This is a reward, remember.

        return tt.sigmoid(x)

    def evaluate_action(self, sa):
        state, action = map(to_tensor, sa)
        s = self.drop_relu(self.input(state))
        x = self.drop_relu(self.fc1(tt.cat([s, action])))
        x = self.output(x)  # Again, activation function here? This is a reward, remember.

        return tt.sigmoid(x)