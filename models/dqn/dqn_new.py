from typing import List, Dict, Union, Tuple
import numpy as np
from loguru import logger
from tqdm import tqdm
import pickle
from models.base_interviewer import InterviewerBase
from models.dqn.dqn_agent import DqnAgent
from models.dqn.dqn_environment import Environment, Rewards
from recommenders.base_recommender import RecommenderBase
from shared.meta import Meta
from shared.user import WarmStartUser
from shared.utility import get_combinations
from experiments.metrics import ndcg_at_k


def choose_candidates(ratings: Dict[int, WarmStartUser], n=100):
    entity_ratings = {}

    for user, data in ratings.items():
        for entity, rating in data.training.items():
            if entity not in entity_ratings:
                entity_ratings[entity] = 0
            if rating > 0:
                entity_ratings[entity] += 1

    sorted_entity_ratings = sorted(entity_ratings.items(), key=lambda x: x[1], reverse=True)
    return [e for e, r in sorted_entity_ratings][:n]


def recent_mean(lst, recency=100):
    if l := len(lst) == 0:
        return 0.0
    recent = lst[l - recency:] if l >= recency else lst
    return np.mean(recent)


def get_rating_matrix(training, n_users, n_entities):
    """
    Returns an [n_users x n_entities] ratings matrix.
    """

    R = np.zeros((n_users, n_entities), dtype=np.float32)
    for user, data in training.items():
        for entity, rating in data.training.items():
            R[user, entity] = rating

    return R


def update_state(state, question, answer):
    s = state.copy()
    entity_idx = question * 2
    s[entity_idx] = 1
    s[entity_idx + 1] = answer

    return s


def get_ndcg(scores, positive_sample):
    ranked_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    relevance_list = [1 if e == positive_sample else 0 for e, s in ranked_scores]
    ndcg = ndcg_at_k(relevance_list, k=10)
    return ndcg


class DqnInterviewer(InterviewerBase):
    """
    A recommendation interviewer that uses reinforcement learning to learn
    how to interview users. Utilises an underlying recommendation model
    to generate rewards.

    Note that the models used with a DQN recommender must be user-agnostic.
    The models should be able to generate recommendations only given the raw
    ratings of a new user, and nothing more. Latent models like MF will have to
    learn a new embedding for every such user.
    """

    def __init__(self, meta: Meta, recommender, use_cuda: bool):
        super(DqnInterviewer, self).__init__(meta)

        self.use_cuda = use_cuda
        self.params = {}

        self.recommender: RecommenderBase = recommender(meta)

        # Allocate DQN agent
        self.agent: Union[DqnAgent, None] = None

        self.n_users = len(self.meta.users)
        self.n_entities = len(self.meta.entities)
        self.candidates = []
        self.ratings: np.ndarray = np.zeros((1,))

    def warmup(self, training: Dict[int, WarmStartUser], interview_length=5):
        n_candidates = 100

        self.candidates = choose_candidates(training, n=n_candidates)
        self.ratings = get_rating_matrix(training, self.n_users, self.n_entities)
        self.recommender.fit(training)

        # NOTE: Test PPR with uniform LOO sampling on training items
        self.test_ppr(training)

        best_agent, best_score, best_params = None, 0, None

        if not self.params:
            for params in get_combinations({'fc1_dims': [128, 256, 512]}):
                agent, score = self.fit(training, interview_length, params)
                if score > best_score:
                    best_agent = agent
                    best_params = params

            self.params = best_params
            self.agent = best_agent
        else:
            logger.info(f'Reusing params for DQN: {self.params}')
            self.agent, _ = self.fit(training, interview_length, self.params)

    def test_ppr(self, training: Dict[int, WarmStartUser]):
        ndcgs = []
        for user, _ in training.items():
            positive_samples, = np.where(self.ratings[user] == 1)
            negative_samples, = np.where(self.ratings[user] == 0)
            left_out_item = np.random.choice(positive_samples)

            # NOTE: self.candidates are the top-100 popular movies
            to_rate = np.random.choice(negative_samples, 100).tolist() + [left_out_item]
            # to_rate = self.candidates + [left_out_item]

            scores = self.recommender.predict(items=to_rate, answers={})
            ndcg = get_ndcg(scores, left_out_item)

            ndcgs.append(ndcg)

        logger.info(f'PPR model NDCG@10 on training items vs. top 100 popular movies: {recent_mean(ndcgs)}')

    def fit(self, training: Dict[int, WarmStartUser], interview_length: int, params: Dict):
        best_agent = None
        best_score = 0

        state_size = len(self.candidates) * 2

        losses = []
        epsilons = []
        ranking_scores = []
        interview_lengths = []

        n_iterations = 50

        self.agent = DqnAgent(candidates=self.candidates, n_entities=self.n_entities,
                              batch_size=720, alpha=0.001, gamma=1.0, epsilon=1.0,
                              eps_end=0.1, eps_dec=0.9996, fc1_dims=params['fc1_dims'], use_cuda=self.use_cuda)

        self.agent.Q_eval.eval()
        score = self.validate(training, interview_length)
        logger.info(f'Start validation score: {score}')
        self.agent.Q_eval.train()

        for iteration in range(n_iterations):
            t = tqdm(training.items())
            for user, _ in t:
                positive_ratings, = np.where(self.ratings[user] == 1)
                negative_ratings, = np.where(self.ratings[user] == 0)

                if not positive_ratings.any():
                    logger.debug(f'Skipping user {user}, no positive ratings')
                    continue

                t.set_description(
                    f'Iteration {iteration} '
                    f'(Ranking Scores: {recent_mean(ranking_scores) : 0.4f}, '
                    f'Loss: {recent_mean(losses) : 0.7f}, '
                    f'Epsilon: {recent_mean(epsilons) : 0.4f}, '
                    f'avg. interview length: {recent_mean(interview_lengths) : 0.4f})')

                # Fresh state
                state = np.zeros((state_size,), dtype=np.float32)
                answers = {}

                # Pick a positive sample and negative samples
                positive_sample = np.random.choice(positive_ratings)
                negative_samples = np.random.choice(negative_ratings, 100).tolist()
                # negative_samples = self.candidates
                to_rate = negative_samples + [positive_sample]
                self.ratings[user, positive_sample] = 0.0

                transitions = []

                for q in range(interview_length):
                    # Find the question to ask
                    question_idx = self.agent.choose_action(state, [])

                    # Ask the question
                    answer = self.ratings[user, self.candidates[question_idx]]

                    # Update state and save the answers
                    new_state = update_state(state, question_idx, answer)
                    answers[self.candidates[question_idx]] = answer
                    state = new_state.copy()

                    # Store transition
                    is_done = q == interview_length - 1
                    transitions.append((state.copy(), question_idx, new_state.copy(), is_done))

                # Rank the to_rate, get the NDCG as the reward
                scores = self.recommender.predict(items=to_rate, answers=answers)
                reward = get_ndcg(scores, positive_sample)

                # Store the transitions in the agent memory
                for (state, question, new_state, is_done) in transitions:
                    self.agent.store_memory(state, question, new_state, reward / interview_length, is_done)

                # Learn from the memories
                loss = self.agent.learn()

                epsilons.append(self.agent.epsilon)
                ranking_scores.append(reward)
                losses.append(loss.cpu().detach().numpy()) if loss is not None else 0.0
                interview_lengths.append(len(answers))

                # Reset the user's rating
                self.ratings[user, positive_sample] = 1.0

            self.agent.Q_eval.eval()
            score = self.validate(training, interview_length)
            logger.info(f'Validation score: {score}')
            self.agent.Q_eval.train()

        return best_agent, best_score

    def validate(self, training: Dict[int, WarmStartUser], interview_length):
        # Validates the performance of a DqnAgent and returns the validation score

        rankings = []

        state_size = len(self.candidates) * 2

        for user, data in training.items():
            state = np.zeros((state_size,), dtype=np.float32)
            answers = {}

            for q in range(interview_length):
                # Ask questions, disregard the reward
                question_idx = self.agent.choose_action(state, [], explore=False)

                # Get the answer
                answer = self.ratings[user, self.candidates[question_idx]]
                answers[self.candidates[question_idx]] = answer

                # Update the state
                new_state = update_state(state, question_idx, answer)
                state = new_state.copy()

            # Get the reward
            scores = self.recommender.predict(data.validation.to_list(), answers)
            rankings.append((data.validation, scores))

        return self.meta.validator.score(rankings, self.meta)

    def get_parameters(self):
        return self.params

    def load_parameters(self, params):
        self.params = params
