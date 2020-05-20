from collections import defaultdict
from typing import List, Dict, Union, Tuple

import dgl
from dgl import DGLHeteroGraph
from loguru import logger
from tqdm import tqdm

from models.base_interviewer import InterviewerBase
from models.dqn.dqn_new import get_reward
from models.gcqn.gcqn import HeteroRGCN
from models.gcqn.gcqn_agent import GcqnAgent
from recommenders.base_recommender import RecommenderBase
from shared.meta import Meta
from shared.user import WarmStartUser
import numpy as np
import torch as tt


def construct_graph(meta: Meta) -> Tuple[DGLHeteroGraph, int]:
    relation_pairs = defaultdict(list)

    for triple in meta.triples:
        h = meta.uri_idx[triple.head]
        t = meta.uri_idx[triple.tail]
        r = triple.relation

        relation_pairs[r].append((h, t))

    relations_dict = {
        ('ENTITY', r, 'ENTITY'): relation_pairs[r]
        for r in relation_pairs.keys()
    }

    u_idx = max(meta.uri_idx.values()) + 1
    relations_dict['USER', 'LIKES', 'ENTITY'] = [(u_idx, 0)]
    relations_dict['USER', 'DISLIKES', 'ENTITY'] = [(u_idx, 1)]

    G = dgl.heterograph(relations_dict)

    return G, u_idx


def get_positive_sample(ratings, user):
    positive_samples, = np.where(ratings[user] == 1)
    return None if not len(positive_samples) else np.random.choice(positive_samples)


def get_negative_samples(ratings, user, n=100):
    negative_samples, = np.where(ratings[user] == 0)
    np.random.shuffle(negative_samples)
    return negative_samples[:n]


def construct_relation_dict(meta: Meta):
    relation_pairs = defaultdict(list)

    for triple in meta.triples:
        h = meta.uri_idx[triple.head]
        t = meta.uri_idx[triple.tail]
        r = triple.relation

        relation_pairs[r].append((h, t))

    relations_dict = {
        ('ENTITY', r, 'ENTITY'): relation_pairs[r]
        for r in relation_pairs.keys()
    }

    u_idx = max(meta.uri_idx.values()) + 1
    return relations_dict, u_idx


def construct_rating_matrix(training: Dict[int, WarmStartUser], n_users: int, n_entities: int):
    ratings = np.zeros((n_users, n_entities))
    for u, data in training.items():
        for entity, rating in data.training.items():
            ratings[u, entity] = rating

    return ratings


def update_relations_dict(relations_dict, user_edges):
    for relation_type, edges in user_edges.items():
        relations_dict[relation_type] = edges

    return relations_dict


class GcqnInterviewer(InterviewerBase):
    def __init__(self, meta: Meta, recommender, use_cuda=False):
        super(GcqnInterviewer, self).__init__(meta)

        self.recommender = recommender(meta)
        self.agent: Union[GcqnAgent, None] = None
        self.G: Union[DGLHeteroGraph, None] = None

        self.n_users = len(meta.users)
        self.n_entities = len(meta.entities)
        self.u_idx = -1

    def warmup(self, training: Dict[int, WarmStartUser], interview_length=5) -> None:
        relation_dict, u_idx = construct_relation_dict(self.meta)
        ratings = construct_rating_matrix(training, self.n_users, self.n_entities)

        self.interview_length = interview_length

        G = dgl.heterograph(relation_dict)

        logger.info(f'Finding candidates...')
        candidates = self.meta.get_question_candidates(training, limit=100)

        logger.info(f'Warming up recommender...')
        self.recommender.parameters = {
            'alpha': 0.1,
            'importance': {1: 0.99, 0: 0.01, -1: 0.0}
        }
        self.recommender.fit(training)

        self.agent = GcqnAgent(
            G=G,
            action_size=self.n_entities,
            memory_size=10000, batch_size=64,
            interview_length=interview_length, n_entities=self.n_entities, candidates=candidates)

        for epoch in range(interview_length):

            losses = []
            rewards = []

            for user, data in tqdm(training.items(), desc=f'[Training on users, epoch {epoch}]'):
                transitions = []

                liked = np.array([0 for _ in range(interview_length)])
                unknown = np.array([0 for _ in range(interview_length)])
                disliked = np.array([0 for _ in range(interview_length)])

                # Pick a positive sample, pick 100 negative samples
                positive_sample = get_positive_sample(ratings, user)
                negative_samples = get_negative_samples(ratings, user)

                if positive_sample is None:
                    logger.info(f'Skipping user {user}, no positive ratings')

                # Update the ratings matrix so they can't answer for their positive sample
                ratings[user, positive_sample] = 0

                for q in range(interview_length):
                    # Run the graph through the network, get a question
                    question = self.agent.choose_action(np.array([liked]), np.array([unknown]), np.array([disliked]))
                    # Ask the user the question, get the answer and add this as an edge in the graph
                    answer = ratings[user, question]
                    old_liked, old_unknown, old_disliked = liked.copy(), unknown.copy(), disliked.copy()

                    (liked if answer == 1 else disliked if answer == -1 else unknown)[q] = question
                    transitions.append(
                        (old_liked, old_unknown, old_disliked,
                         question,
                         liked, unknown, disliked,
                         q == interview_length - 1))

                # Pass along the answers to the recommender, get reward, store transitions
                answers = {}
                for e in liked:
                    answers[e] = 1
                for e in disliked:
                    answers[e] = -1
                for e in unknown:
                    answers[e] = 0

                to_rate = negative_samples.tolist() + [positive_sample]

                scores = self.recommender.predict(items=to_rate, answers=answers)
                reward = get_reward(scores, positive_sample, answers, interview_length)

                rewards.append(reward)

                for (l, u, d, q, n_l, n_u, n_d, t) in transitions:
                    r = reward if t else 0.0
                    self.agent.store_memory(l, u, d, q, n_l, n_u, n_d, r, t)

                loss = self.agent.learn()
                losses.append(loss.detach().cpu().numpy()) if loss is not None else None

                ratings[user, positive_sample] = 1

            logger.info(f'Rewards: {np.mean(rewards)}')
            logger.info(f'Loss: {np.mean(losses)}')
            logger.info(f'Epsilon: {self.agent.epsilon}')

    def interview(self, answers: Dict, max_n_questions=5) -> List[int]:
        liked = np.array([0 for _ in range(self.interview_length)])
        unknown = np.array([0 for _ in range(self.interview_length)])
        disliked = np.array([0 for _ in range(self.interview_length)])

        for i, (e, a) in enumerate(answers.items()):
            if a == 1:
                liked[i] = e
            elif a == -1:
                disliked[i] = e
            else:
                unknown[i] = e

        with tt.no_grad():
            question = self.agent.choose_action(np.array([liked]), np.array([unknown]), np.array([disliked]),
                                                explore=False)

        return [question]

    def predict(self, items: List[int], answers: Dict) -> Dict[int, float]:
        return self.recommender.predict(items=items, answers=answers)

    def get_parameters(self):
        pass

    def load_parameters(self, params):
        pass