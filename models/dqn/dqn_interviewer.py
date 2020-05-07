from typing import List, Dict, Union, Tuple
import numpy as np
from loguru import logger
from tqdm import tqdm
import pickle
from models.base_interviewer import InterviewerBase
from models.dqn.dqn_agent import DqnAgent
from models.dqn.dqn_environment import Environment, Rewards
from models.greedy.greedy_interviewer import GreedyInterviewer
from recommenders.base_recommender import RecommenderBase
from shared.meta import Meta
from shared.user import WarmStartUser
from shared.utility import get_combinations, get_top_entities


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


def recent_mean(lst, recency=10):
    if l := len(lst) == 0:
        return 0.0
    recent = lst[l - recency:] if l >= recency else lst
    return np.mean(recent)


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

        # Allocate environment
        if not recommender:
            raise RuntimeError('No underlying recommender provided to the naive interviewer.')

        self.recommender: RecommenderBase = recommender(meta)
        self.environment: Union[Environment, None] = None

        # Allocate DQN agent
        self.agent: Union[DqnAgent, None] = None

        self.n_users = len(self.meta.users)
        self.n_entities = len(self.meta.entities)
        self.candidates = []

    def _choose_candidates(self, training, n):
        greedy_interviewer = GreedyInterviewer(self.meta, self.recommender)

        # To speed up things, consider only the informativeness of most popular n * 2 entities
        entity_scores = greedy_interviewer.get_entity_scores(training, get_top_entities(training)[:n * 2], list())

        # Select n most informative entities as candidates
        return [entity for entity, score in entity_scores[:n]]

    def warmup(self, training: Dict[int, WarmStartUser], interview_length=5):
        n_candidates = 20

        logger.info(f'DQN warming up environment...')
        self.environment = Environment(
            recommender=self.recommender, reward_metric=Rewards.NDCG, meta=self.meta, state_size=n_candidates)
        self.environment.warmup(training)
        self.candidates = self._choose_candidates(training, n=n_candidates)

        best_agent, best_score, best_params = None, 0, None

        if not self.params:
            for params in get_combinations({'fc1_dims': [128, 256, 512]}):
                agent, score = self.train_dqn(training, interview_length, params)
                if score > best_score:
                    best_agent = agent
                    best_params = params

            self.params = best_params
            self.agent = best_agent
        else:
            logger.info(f'Reusing params for DQN: {self.params}')
            self.agent, _ = self.train_dqn(training, interview_length, self.params)

    def train_dqn(self, training: Dict[int, WarmStartUser], interview_length: int, params: Dict):
        # Trains k different DqnAgents and returns the best performing one
        n_agents = 10
        best_agent = None
        best_score = 0

        for n in range(n_agents):
            self.agent = DqnAgent(candidates=self.candidates, n_entities=self.n_entities,
                                  batch_size=64, alpha=0.0003, gamma=1.0, epsilon=1.0,
                                  eps_end=0.1, eps_dec=0.996, fc1_dims=params['fc1_dims'], use_cuda=self.use_cuda)

            logger.info(f'Training {n+1} of {n_agents} agents...')
            agent = self.fit_dqn(training, interview_length)
            score = self.validate_dqn(training, interview_length)

            logger.info(f'Validated with score {score} ({self.meta.validator.metric})')

            if score > best_score:
                logger.info(f'Found new best DQN with params {params}, score {score}')
                best_agent = pickle.loads(pickle.dumps(agent))
                best_score = score

        return best_agent, best_score

    def validate_dqn(self, training: Dict[int, WarmStartUser], interview_length: int) -> float:
        # Validates the performance of a DqnAgent and returns the validation score

        rankings = []

        for user, data in training.items():
            state = self.environment.reset()

            self.environment.select_user(user, remove_positive_sample=False)

            for q in range(interview_length):
                # Ask questions, disregard the reward
                question = self.agent.choose_action(state)
                new_state, _ = self.environment.ask(user, question, self.candidates[question])
                state = new_state

            # Send the state to the environment's recommender along with user.validation, get the ranking
            ranking = self.environment.rank(data.validation.to_list())
            rankings.append((data.validation, ranking))

        # Return the meta.validator() score
        return self.meta.validator.score(rankings, self.meta)

    def fit_dqn(self, training: Dict[int, WarmStartUser], interview_length: int) -> DqnAgent:
        n_iterations = 10

        epsilons = []
        scores = []
        losses = []

        self.agent.Q_eval.train()

        for i in range(n_iterations):

            users = list(training.keys())
            np.random.shuffle(users)

            t = tqdm(users)
            for user in t:

                ps, = np.where(self.environment.ratings[user] > 0)
                if not ps.any():
                    logger.debug('Skipping user with no positive ratings')
                    continue

                t.set_description(f'Iteration {i} (Scores: {recent_mean(scores) : 0.4f}, Loss: {recent_mean(losses) : 0.7f}, '
                                  f'Epsilon: {recent_mean(epsilons) : 0.4f})')

                state = self.environment.reset()

                pos_ratings, = np.where(self.environment.ratings[user] == 1)
                if len(pos_ratings) == 0:
                    logger.debug(f'Skipping user {user} due to no positive ratings')
                    continue

                self.environment.select_user(user)

                # Train on the memories
                _loss = self.agent.learn()
                _reward = 0

                for q in range(interview_length):
                    question = self.agent.choose_action(state)
                    new_state, reward = self.environment.ask(user, question, self.candidates[question])

                    # Store the memory for training
                    self.agent.store_memory(state, question, new_state, reward, q == interview_length)
                    state = new_state

                    _reward += reward

                # Record scores, losses, rewards
                epsilons.append(self.agent.epsilon)
                scores.append(_reward)
                if _loss is not None:
                    losses.append(_loss.cpu().detach().numpy())

        self.agent.Q_eval.eval()
        return self.agent

    def interview(self, answers: Dict, max_n_questions=5) -> List[int]:
        state = np.zeros((len(self.candidates) * 2,), dtype=np.float32)
        for entity, rating in answers.items():
            entity_state_idx = self.candidates.index(entity) * 2
            state[entity_state_idx] = 1
            state[entity_state_idx + 1] = rating

        question_idx = self.agent.choose_action(state, explore=False)
        return [self.candidates[question_idx]]

    def predict(self, items: List[int], answers: Dict) -> Dict[int, float]:
        return self.environment.recommender.predict(items, answers)

    def get_parameters(self):
        return self.params

    def load_parameters(self, params):
        self.params = params
