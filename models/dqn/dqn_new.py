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
from shared.enums import Sentiment
from shared.meta import Meta
from shared.user import WarmStartUser
from shared.utility import get_combinations, get_top_entities
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


def get_hit(scores, positive_sample):
    ranked_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    relevance_list = [1 if e == positive_sample else 0 for e, s in ranked_scores]
    return np.sum(relevance_list[:10])


def get_ndcg(scores, positive_sample):
    ranked_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    relevance_list = [1 if e == positive_sample else 0 for e, s in ranked_scores]
    ndcg = ndcg_at_k(relevance_list, k=10)
    return ndcg


def get_reward(scores, positive_sample, answers, interview_length):
    # if len(answers) < interview_length:
    #     return 0.0
    return get_ndcg(scores, positive_sample)
    # return get_hit(scores, positive_sample)


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

    def _choose_candidates(self, training, n):
        greedy_interviewer = GreedyInterviewer(self.meta, self.recommender)

        # To speed up things, consider only the informativeness of most popular n * 2 entities
        entity_scores = greedy_interviewer.get_entity_scores(training, get_top_entities(training)[:n * 2], list())

        # Select n most informative entities as candidates
        return [entity for entity, score in entity_scores[:n]]

    def warmup(self, training: Dict[int, WarmStartUser], interview_length=5):
        n_candidates = 20

        self.ratings = get_rating_matrix(training, self.n_users, self.n_entities)
        self.recommender.fit(training)
        self.candidates = self.meta.get_question_candidates(training, limit=n_candidates)

        # NOTE: Test PPR with uniform LOO sampling on training items

        best_agent, best_score, best_params = None, 0, None

        self.params = {'fc1_dims': 512}  # Don't tune hyper params, we don't have time

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

    def test_naive(self, training: Dict[int, WarmStartUser]):
        candidate_scores = {}

        for candidate in tqdm(self.candidates, desc="Looking at candidates..."):
            predictions = []
            for user, data in training.items():
                answers = {candidate: self.ratings[user, candidate]}
                scores = self.recommender.predict(data.validation.to_list(), answers)
                predictions.append((data.validation, scores))

            candidate_scores[candidate] = self.meta.validator.score(predictions, self.meta)

        sorted_candidates = list(sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True))
        logger.info(f'Best candidate: {sorted_candidates[0]}')
        logger.info(f'Worst candidate: {sorted_candidates[-1]}')

    def test_ppr(self, training: Dict[int, WarmStartUser]):
        ndcgs = []
        random_ndcgs = []

        for user, data in training.items():
            positive_samples, = np.where(self.ratings[user][self.meta.recommendable_entities] == 1)
            negative_samples, = np.where(self.ratings[user][self.meta.recommendable_entities] == 0)

            if not positive_samples.any() or len(negative_samples) < 100:
                logger.debug(f'Skipping user {user} due to not enough ratings')
                continue

            left_out_item = np.random.choice(positive_samples)

            # NOTE: self.candidates are the top-100 popular movies
            np.random.shuffle(negative_samples)
            to_rate = negative_samples[:100].tolist() + [left_out_item]
            # to_rate = self.candidates + [left_out_item]

            scores = self.recommender.predict(items=to_rate, answers={})
            random_scores = {e: np.random.random() for e in to_rate}
            ndcg = get_ndcg(scores, left_out_item)
            random_ndcg = get_ndcg(random_scores, left_out_item)

            ndcgs.append(ndcg)
            random_ndcgs.append(random_ndcg)

        logger.info(f'PPR model NDCG@10 on training items vs. top 100 popular movies: {np.mean(ndcgs)} (random is {np.mean(random_ndcgs)})')

    def fit(self, training: Dict[int, WarmStartUser], interview_length: int, params: Dict):
        best_agent = None
        best_score = 0

        state_size = len(self.candidates) * 2

        losses = []
        epsilons = []
        ranking_scores = []
        corrects = []
        interview_lengths = []
        interview_answers = {
            1: [], 0: [], -1: []
        }

        n_iterations = 20

        self.agent = DqnAgent(candidates=self.candidates, n_entities=self.n_entities,
                              batch_size=720, alpha=0.0001, gamma=1.0, epsilon=1.0,
                              eps_end=0.1, eps_dec=0.996, fc1_dims=params['fc1_dims'], use_cuda=self.use_cuda,
                              interview_length=interview_length)

        self.agent.Q_eval.eval()
        score = self.validate(training, interview_length)
        logger.info(f'Start validation score: {score}')
        self.agent.Q_eval.train()

        last_iteration_loss = np.inf

        for iteration in range(n_iterations):
            users = list(training.keys())
            np.random.shuffle(users)
            t = tqdm(users)
            pass_count = 0
            for user in t:
                positive_ratings, = np.where(self.ratings[user][self.meta.recommendable_entities] == 1)
                negative_ratings, = np.where(self.ratings[user][self.meta.recommendable_entities] == 0)

                if not positive_ratings.any() or len(negative_ratings) < 100:
                    continue

                t.set_description(
                    f'Iteration {iteration} '
                    f'(Ranking Scores: {recent_mean(ranking_scores) : 0.4f}, '
                    f'Loss: {recent_mean(losses) : 0.7f}, '
                    f'Epsilon: {recent_mean(epsilons) : 0.4f}, '
                    f'avg. interview length: {recent_mean(interview_lengths) : 0.4f}) '
                    f'Like: {recent_mean(interview_answers[1]) : 0.3f}, Dislike: {recent_mean(interview_answers[-1]) : 0.3f}, Dunno: {recent_mean(interview_answers[0]) : 0.3f}')

                # Fresh state
                state = np.zeros((state_size,), dtype=np.float32)
                answers = {}

                # Pick a positive sample and negative samples
                positive_sample = np.random.choice(positive_ratings)
                np.random.shuffle(negative_ratings)
                negative_samples = negative_ratings[:10].tolist()
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
                    transitions.append((state.copy(), question_idx, answer, new_state.copy(), is_done))

                    if answer > 0:
                        interview_answers[1].append(1)
                        interview_answers[0].append(0)
                        interview_answers[-1].append(0)
                    elif answer < 0:
                        interview_answers[1].append(0)
                        interview_answers[0].append(0)
                        interview_answers[-1].append(1)
                    else:
                        interview_answers[1].append(0)
                        interview_answers[0].append(1)
                        interview_answers[-1].append(0)

                # Rank the to_rate, get the NDCG as the reward
                # positive_sample, = training[user].validation.sentiment_samples[Sentiment.POSITIVE]
                # to_rate = training[user].validation.to_list()
                scores = self.recommender.predict(items=to_rate, answers=answers)
                reward = get_reward(scores, positive_sample, answers, interview_length)
                # reward = self.meta.validator.score([(training[user].validation, scores)], self.meta)

                # Store the transitions in the agent memory
                for state, question, answer, new_state, is_done in transitions:
                    _reward = reward * 100 if is_done else reward / interview_length
                    self.agent.store_memory(state, question, new_state, _reward, is_done)

                # Learn from the memories
                loss = self.agent.learn()

                pass_count += 1

                epsilons.append(self.agent.epsilon)
                ranking_scores.append(reward)
                losses.append(loss.cpu().detach().numpy()) if loss is not None else 0.0
                interview_lengths.append(len(answers))

                # Reset the user's rating
                self.ratings[user, positive_sample] = 1.0

            self.agent.Q_eval.eval()
            score = self.validate(training, interview_length)
            if score > best_score:
                best_agent = pickle.loads(pickle.dumps(self.agent))
                best_score = score
                logger.info(f'Found new best agent with validation score {score}')
            self.agent.Q_eval.train()

            # iteration_loss = recent_mean(losses, recency=len(users))
            # if iteration_loss > last_iteration_loss:
            #     logger.info(f'Stopping due to increasing loss')
            #     return best_agent, best_score
            # else:
            #     last_iteration_loss = iteration_loss

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

    def interview(self, answers: Dict, max_n_questions=5) -> List[int]:
        questions = []
        state = np.zeros((len(self.candidates) * 2,), dtype=np.float32)
        for entity, answer in answers.items():
            entity_idx = self.candidates.index(entity)
            state = update_state(state, entity_idx, answer)

        question_idx = self.agent.choose_action(state, questions, explore=False)
        questions.append(question_idx)
        return [self.candidates[question_idx]]

    def predict(self, items: List[int], answers: Dict) -> Dict[int, float]:
        return self.recommender.predict(items, answers)

    def get_parameters(self):
        return self.params

    def load_parameters(self, params):
        self.params = params
