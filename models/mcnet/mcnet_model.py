import torch as tt
import torch.nn as nn
import torch.nn.functional as ff


def hinge_loss(positive, negative):
    return (tt.relu(1 - (positive - negative)) ** 2).sum()


def logistic_ranking_loss(positive, negative):
    return tt.log(1 + tt.exp(-(positive - negative)))


class MonteCarloNet(nn.Module):
    def __init__(self, input_dims, hidden_dims, method):
        super(MonteCarloNet, self).__init__()

        self.fc1 = nn.Linear(input_dims, hidden_dims)
        self.fc2 = nn.Linear(hidden_dims, hidden_dims)
        self.fc3 = nn.Linear(hidden_dims, input_dims)

        self.device = tt.device('cuda' if tt.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.prediction_activation = tt.sigmoid if method == 'pairwise' else tt.tanh

    def forward(self, user_ratings, dropout=True):
        user_ratings = tt.tensor(user_ratings).to(self.device)

        def apply_dropout(input, dropout):
            p = 0.5
            return ff.dropout(input, p=p) if dropout else input

        x = tt.relu(apply_dropout(self.fc1(user_ratings), dropout))
        x = tt.relu(apply_dropout(self.fc2(x), dropout))
        x = self.prediction_activation(apply_dropout(self.fc3(x), dropout))

        return x
