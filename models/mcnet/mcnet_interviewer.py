import pickle
import random
from typing import List, Dict, Union

from loguru import logger

from models.base_interviewer import InterviewerBase
from models.mcnet.mcnet_model import MonteCarloNet, hinge_loss, logistic_ranking_loss
from shared.meta import Meta
from shared.user import WarmStartUser
import numpy as np
import torch as tt
import torch.nn as nn
import matplotlib.pyplot as plt


def get_rating_matrix(training: Dict[int, WarmStartUser], n_users, n_entities):
    R = np.zeros((n_users, n_entities))
    for u, data in training.items():
        for entity, rating in data.training.items():
            R[u, entity] = rating

    return R


def get_log_popularities(rating_matrix: np.ndarray):
    log_popularities = np.zeros((rating_matrix.shape[1]))
    for e, entity_rating_vector in enumerate(rating_matrix.T):
        ratings, = np.where(entity_rating_vector != 0)
        n_ratings = len(ratings)
        # log_popularities[e] = np.log(n_ratings) if n_ratings else 0.0
        log_popularities[e] = n_ratings

    return log_popularities


class MonteCarloNetInterviewer(InterviewerBase):
    def __init__(self, meta: Meta, use_cuda=False):
        super(MonteCarloNetInterviewer, self).__init__(meta)
        self.n_users = len(self.meta.users)
        self.n_entities = len(self.meta.entities)
        self.model: Union[MonteCarloNet, None] = None
        self.idx_uri_map = self.meta.get_idx_uri()
        self.log_popularities = np.zeros((self.n_entities,))

        self.method = 'pairwise'
        self.loss_functions = {
            'pairwise': logistic_ranking_loss,
            'prediction': nn.MSELoss()
        }

    def warmup(self, training: Dict[int, WarmStartUser], interview_length=5) -> None:
        if self.method == 'pairwise':
            self.warmup_pairwise(training, interview_length)
        elif self.method == 'prediction':
            self.warmup_prediction(training, interview_length)

    def warmup_pairwise(self, training: Dict[int, WarmStartUser], interview_length=5) -> None:
        R = get_rating_matrix(training, self.n_users, self.n_entities)
        self.log_popularities = get_log_popularities(R)

        self.model = MonteCarloNet(self.n_entities, hidden_dims=512, method=self.method)

        loss_fnc = self.loss_functions[self.method]
        optimizer = tt.optim.Adam(self.model.parameters(), lr=0.001)
        best_model, best_score = None, 0

        patience_counter = 0
        max_patience = 10

        epoch_losses = []
        epoch_scores = []

        for epoch in range(100):
            epoch_loss = tt.tensor(0.0).to(self.model.device)
            for user_ratings_vector in R:
                # Copy the vector so we don't persist changes into the ratings matrix
                user_ratings_vector = user_ratings_vector.copy()
                # Pick, at random, one positive and one negative sample from the
                # user's ratings.
                positive_ratings, = np.where(user_ratings_vector == 1)
                unseen_ratings, = np.where(user_ratings_vector == 0)

                if not len(positive_ratings):
                    continue
                pos = np.random.choice(positive_ratings)
                neg = np.random.choice(unseen_ratings)

                user_ratings_vector[pos] = 0

                # Construct the user vector from the remaining ratings.
                # user_vector = np.zeros((self.n_entities,), dtype=np.float32)
                # user_vector[positive_ratings] = user_ratings_vector[remaining_entities]
                user_vector = user_ratings_vector.astype(np.float32)
                # Pass the user vector through the network and receive scores
                # for all items.
                scores = self.model(user_vector)
                # Calculate hinge loss (or pairwise ranking loss)
                # for the positive and negative sample we drew before.
                loss = loss_fnc(scores[pos], scores[neg])
                # Accumulate loss for back propagation and repeat.
                epoch_loss += loss

            self.model.zero_grad()
            epoch_loss.backward()
            optimizer.step()

            validation_score = self.validate(training, R)
            logger.info(f'Epoch {epoch}: {epoch_loss / self.n_users} - validation score: {validation_score}')

            epoch_losses.append(epoch_loss / self.n_users)
            epoch_scores.append(validation_score * 10)

            if validation_score > best_score:
                best_model = pickle.loads(pickle.dumps(self.model))
                best_score = validation_score
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter > max_patience:
                    logger.info(f'Stopping training due to non-increasing validation for {max_patience} epochs.')
                    break

        logger.info(f'Using best model with validation score: {best_score}')
        self.model = best_model

        plt.plot(epoch_losses, label='Hinge loss f(x, y) = max(0, 1 - (x - y))^2')
        plt.plot(epoch_scores,
                 label=f'Validation '
                       f'{self.meta.validator.metric}@{self.meta.validator.cutoff}'
                       f' (scaled by 10 for comparison)')
        plt.xlabel('Epochs')
        plt.legend()
        plt.show()

    def warmup_prediction(self, training: Dict[int, WarmStartUser], interview_length=5) -> None:
        R = get_rating_matrix(training, self.n_users, self.n_entities)
        self.log_popularities = get_log_popularities(R)

        self.model = MonteCarloNet(self.n_entities, hidden_dims=512, method=self.method)

        loss_fnc = tt.nn.MSELoss()
        optimizer = tt.optim.Adam(self.model.parameters(), lr=0.001)

        best_model, best_score = None, 0

        epoch_losses = []
        epoch_scores = []

        for epoch in range(100):
            epoch_loss = tt.tensor(0.0).to(self.model.device)
            for user_ratings_vector in R:
                # Get the user's ratings (likes and dislikes)
                rated_entities, = np.where(user_ratings_vector != 0)
                rated_entities = list(rated_entities)
                # Pick 50% of them for training, 50% as labels
                n_rated_entities = len(rated_entities)
                test_entities = random.choices(population=rated_entities, k=n_rated_entities // 2)
                train_entities = [e for e in rated_entities if e not in test_entities]

                if not test_entities or not train_entities:
                    continue
                # Construct training and label vectors
                train_vector = np.zeros((self.n_entities,), dtype=np.float32)
                test_vector = np.zeros((self.n_entities,), dtype=np.float32)
                train_vector[train_entities] = user_ratings_vector[train_entities]
                test_vector[test_entities] = user_ratings_vector[test_entities]
                # Pass through network and get the predicted ratings
                predictions = self.model(train_vector)

                # Calculate loss (MSE or Cross Entropy), back propagate and repeat.
                loss = loss_fnc(predictions[test_entities],
                                tt.tensor(user_ratings_vector[test_entities], dtype=tt.float32).to(self.model.device))

                epoch_loss += loss

            self.model.zero_grad()
            epoch_loss.backward()
            optimizer.step()

            validation_score = self.validate(training, R)
            logger.info(f'Epoch {epoch}: {epoch_loss / self.n_users} - validation score: {validation_score}')
            epoch_losses.append(epoch_loss / self.n_users)
            epoch_scores.append(validation_score * 10)

            if validation_score > best_score:
                best_model = pickle.loads(pickle.dumps(self.model))
                best_score = validation_score

        logger.info(f'Using best model with validation score: {best_score}')
        self.model = best_model

        plt.plot(epoch_losses, label='MSE loss')
        plt.plot(epoch_scores,
                 label=f'Validation '
                       f'{self.meta.validator.metric}@{self.meta.validator.cutoff}'
                       f' (scaled by 10 for comparison)')
        plt.xlabel('Epochs')
        plt.legend()
        plt.show()

    def validate(self, training: Dict[int, WarmStartUser], rating_matrix: np.ndarray):
        predictions = []
        self.model.eval()
        for user, data in training.items():
            # Get their ratings from the rating matrix, construct the training vector
            user_ratings_vector = rating_matrix[user]
            rated_entities, = np.where(user_ratings_vector != 0)
            user_vector = np.zeros((self.n_entities,), dtype=np.float32)
            user_vector[rated_entities] = user_ratings_vector[rated_entities]
            # Then, pass through the model and receive a score for each entity (their rank)
            scores = self.model(user_vector, dropout=False).detach().cpu().numpy()
            # Get the scores for just the validation items (+ the left out one) and store these
            to_predict = data.validation.to_list()
            predictions.append((data.validation, {item: scores[item] for item in to_predict}))
            # in predictions.
            # Run the meta.validator on the predictions and show the score.

        score = self.meta.validator.score(predictions, self.meta)
        self.model.train()
        return score

    def interview(self, answers: Dict, max_n_questions=5) -> List[int]:
        # Create the user vector and pass it through the network X times.
        # Then, for every entity, calculate the variance.
        # Optionally, scale the variance by popularity.
        # Then, select the question with the largest variance.
        self.model.eval()
        n_predictions = 100
        user_vector = np.zeros((self.n_entities,), dtype=np.float32)
        for entity, answer in answers.items():
            user_vector[entity] = answer

        repeat_user_vector = np.array([user_vector, ] * n_predictions)
        predictions = self.model(repeat_user_vector, dropout=True)
        candidate_entities = [e for e in range(self.n_entities) if e not in answers]
        variances = predictions.var(dim=0).detach().cpu().numpy()
        variances *= self.log_popularities
        uncertain_entity = variances[candidate_entities].argmax()

        return [uncertain_entity]

    def predict(self, items: List[int], answers: Dict) -> Dict[int, float]:
        self.model.eval()
        user_vector = np.zeros((self.n_entities,), dtype=np.float32)
        for entity, answer in answers.items():
            user_vector[entity] = answer

        scores = self.model(user_vector, dropout=False)
        return {item: scores[item] for item in items}

    def get_parameters(self):
        pass

    def load_parameters(self, params):
        pass