import torch as tt
import torch.nn as nn
import torch.nn.functional as ff
import torch.optim as optim
from loguru import logger


class DeepQNetwork(nn.Module):
    def __init__(self, alpha, state_size: int, fc1_dims: int, fc2_dims: int, actions_size: int, interview_length: int, use_cuda: bool = True):
        super(DeepQNetwork, self).__init__()

        self.layers = {
            q: nn.Linear(state_size, fc1_dims)
            for q in range(interview_length)
        }

        # Optimisers and loss
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.loss = ff.smooth_l1_loss

        # Send the model to GPU if possible
        self.device = tt.device('cuda' if tt.cuda.is_available() else 'cpu')
        self.to(self.device)

        logger.info(f'Sent DQN model to device: {self.device}')

    def forward(self, state, question_number):
        state = tt.tensor(state).to(self.device)
        layer = self.layers[question_number]
        return ff.relu(layer(state))

    def get_loss(self, current_predicted_rewards, target_rewards):
        target_rewards = target_rewards.to(self.device)
        # current_predicted_rewards = tt.tensor(current_predicted_rewards).to(self.device)
        # return tt.abs(tt.sum(target_rewards - current_predicted_rewards))
        return self.loss(current_predicted_rewards, target_rewards)
