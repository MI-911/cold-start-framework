import torch as tt
import torch.nn as nn
import torch.nn.functional as ff
import torch.optim as optim


class DeepQNetwork(nn.Module):
    def __init__(self, alpha, state_size: int, fc1_dims: int, fc2_dims: int, actions_size: int, use_cuda: bool = True):
        super(DeepQNetwork, self).__init__()

        # Construct layers
        self.fc1 = nn.Linear(state_size, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, actions_size)

        # Optimisers and loss
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.loss = ff.smooth_l1_loss
        # self.loss = nn.MSELoss()

        # Send the model to GPU if possible
        self.device = tt.device('cuda' if tt.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        state = tt.tensor(state).to(self.device)

        x = ff.relu(self.fc1(state))
        x = ff.relu(self.fc2(x))
        return self.fc3(x)

    def get_loss(self, current_predicted_rewards, target_rewards):
        target_rewards = target_rewards.to(self.device)
        # current_predicted_rewards = tt.tensor(current_predicted_rewards).to(self.device)
        return self.loss(current_predicted_rewards, target_rewards)
