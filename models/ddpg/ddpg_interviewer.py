from typing import List, Dict, Union

from loguru import logger
import numpy as np
from tqdm import tqdm

from experiments.metrics import ndcg_at_k
from models.base_interviewer import InterviewerBase
from models.ddpg.agent import DDPGAgent
from models.ddpg.model import Actor
from models.ddpg.utils import to_tensor
from recommenders.base_recommender import RecommenderBase
from recommenders.mf.mf_recommender import MatrixFactorizationRecommender
from shared.meta import Meta
from shared.user import WarmStartUser


def get_rating_matrix(training, n_users, n_entities):

    R = np.zeros((n_users, n_entities))
    for user, data in training.items():
        for entity, rating in data.training.items():
            R[user, entity] = rating

    return R


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


def get_best_candidate(action_vector, candidate_vectors, critic, state):
    # Find the candidates closest to the action in the latent space
    distances = [(i, ((action_vector - candidate_vector) ** 2).sum()) for i, candidate_vector in enumerate(candidate_vectors)]
    sorted_distances = list(sorted(distances, key=lambda x: x[1], reverse=False))
    top_candidates, min_distances = zip(*sorted_distances[:10])

    # Find the candidate with the highest predicted reward
    states = to_tensor(np.asarray([state for _ in top_candidates], dtype=np.float32))
    actions = to_tensor(np.asarray([candidate_vectors[candidate] for candidate in top_candidates], dtype=np.float32))

    candidate_rewards = critic([states, actions]).squeeze()
    top_candidates, rewards = zip(*list(sorted(zip(top_candidates, candidate_rewards), key=lambda x: x[1], reverse=True)))

    return top_candidates[0]


def update_state(state, question, answer):
    state = state.copy()
    question_idx = question * 2
    state[question_idx] = 1
    state[question_idx + 1] = answer

    return state


def recent_mean(lst, recency=10):
    if l := len(lst) == 0:
        return 0.0
    recent = lst[l - recency:] if l >= recency else lst
    return np.mean(recent)


def has_positive_ratings(user, ratings, items):
    positive_ratings, = np.where(ratings[user][items] > 0)
    return positive_ratings.any()


def get_ndcg(predictions, left_out):
    k = 10
    rankings = [r for r, s in sorted(predictions.items(), key=lambda x: x[1], reverse=True)]
    relevance = [1 if r == left_out else 0 for r in rankings]

    return ndcg_at_k(relevance, k=k)


def sample_for_user(user, ratings, recommendable_entities):
    positive_samples, = np.where(ratings[user][recommendable_entities] == 1)
    negative_samples, = np.where(ratings[user][recommendable_entities] == 0)

    positive_sample = np.random.choice(positive_samples)
    np.random.shuffle(negative_samples)
    negative_samples = negative_samples[:100]

    to_rate = negative_samples + [positive_sample]
    return to_rate, positive_sample


class DDPGInterviewer(InterviewerBase):
    def __init__(self, meta: Meta, recommender, use_cuda: bool):
        super(DDPGInterviewer, self).__init__(meta)

        self.embedder: MatrixFactorizationRecommender = MatrixFactorizationRecommender(meta, normalize_embeddings=True)
        self.recommender: RecommenderBase = recommender(meta)
        self.agent: Union[DDPGAgent, None] = None
        self.candidates = []
        self.ratings = np.zeros((1,))

        self.n_users = len(self.meta.users)
        self.n_entities = len(self.meta.entities)
        self.n_candidates = 100
        self.candidate_embeddings = []
        self.state_size = self.n_candidates * 2

    def warmup(self, training: Dict[int, WarmStartUser], interview_length=5) -> None:

        self.candidates = choose_candidates(training, n=self.n_candidates)
        self.ratings = get_rating_matrix(training, n_users=self.n_users, n_entities=self.n_entities)

        self.embedder.fit(training)
        self.recommender.fit(training)
        self.agent = DDPGAgent(embedding_size=self.embedder.model.k, state_size=self.state_size)

        self.candidate_embeddings = self.embedder.model.M[self.candidates]

        critic_losses = []
        actor_losses = []
        epsilons = []
        rewards = []

        steps_taken = 0

        for iteration in range(50):
            warmup_steps = 2000

            t = tqdm(training.items())
            for user, data in t:
                state = np.zeros((self.state_size,))
                answers = {}
                transitions = []

                # Skip the user if they don't have the ratings we need
                if not has_positive_ratings(user, self.ratings, self.meta.recommendable_entities):
                    continue

                # Sample for the user
                to_rate, positive_sample = sample_for_user(user, self.ratings, self.meta.recommendable_entities)
                self.ratings[user, positive_sample] = 0

                t.set_description(
                    f'[{iteration}][Critic loss: {recent_mean(critic_losses) : 0.4f} '
                    f'Actor loss: {recent_mean(actor_losses) : 0.4f} '
                    f'Rewards: {recent_mean(rewards) : 0.4f} '
                    f'Epsilon: {recent_mean(epsilons) : 0.4f}]'
                )

                for q in range(interview_length):
                    # Get a proto-action vector
                    action = self.agent.choose_action(
                        state) if steps_taken > warmup_steps else self.agent.random_action()

                    # Use the vector to select a question (kNN)
                    candidate = get_best_candidate(action, self.candidate_embeddings, self.agent.critic, state)
                    question = self.candidates[candidate]

                    # Ask the question, receive a new state
                    answer = self.ratings[user, question]
                    new_state = update_state(state, candidate, answer)
                    answers[question] = answer

                    is_done = q == interview_length - 1

                    # Store transition
                    transitions.append((state.copy(), action.copy(), answer, new_state.copy(), is_done))
                    state = new_state

                    # Update the policy
                    if steps_taken > warmup_steps:
                        critic_loss, actor_loss = self.agent.update_policy()
                        critic_losses.append(critic_loss)
                        actor_losses.append(actor_loss)

                # Get the reward for this interview
                predictions = self.recommender.predict(to_rate, answers)
                reward = get_ndcg(predictions, positive_sample)

                # Memorise the transitions with rewards
                for state, action, answer, new_state, is_done in transitions:
                    r = reward * 100 if is_done else 0.0 if not answer else 1.0
                    self.agent.memory.store(state, action, new_state, r, is_done)

                rewards.append(reward)
                epsilons.append(self.agent.epsilon)

                steps_taken += 1

                # Reset user rating
                self.ratings[user, positive_sample] = 1

            if steps_taken > warmup_steps:
                validation_score = self.validate(training, interview_length)
                logger.info(
                    f'Validation ({self.meta.validator.metric}@{self.meta.validator.cutoff}): {validation_score}')

        input()

    def validate(self, training: Dict[int, WarmStartUser], interview_length: int):
        self.agent.eval()

        predictions = []
        for user, data in tqdm(training.items(), desc=f'[Validating]'):
            to_rate = data.validation.to_list()

            state = np.zeros((self.state_size,))
            answers = {}

            for q in range(interview_length):
                # Get a proto-action vector
                action = self.agent.choose_action(state, add_noise=False, decay_epsilon=False)

                # Use the vector to select a question (kNN)
                candidate = get_best_candidate(action, self.candidate_embeddings, self.agent.critic, state)
                question = self.candidates[candidate]

                # Ask the question, receive a new state
                answer = self.ratings[user, question]
                new_state = update_state(state, candidate, answer)
                answers[question] = answer
                state = new_state

            prediction = self.recommender.predict(to_rate, answers)
            predictions.append((data.validation, prediction))

        self.agent.train()
        return self.meta.validator.score(predictions, self.meta)

    def interview(self, answers: Dict, max_n_questions=5) -> List[int]:
        pass

    def predict(self, items: List[int], answers: Dict) -> Dict[int, float]:
        pass

    def get_parameters(self):
        pass

    def load_parameters(self, params):
        pass