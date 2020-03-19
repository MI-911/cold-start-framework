import sys
from collections import OrderedDict, Counter
from copy import deepcopy
from random import shuffle
from typing import Dict, List

from loguru import logger
from torch.autograd import Variable

import torch as tt
from torch.nn import functional as F


import pandas as pd
import numpy as np
from tqdm import tqdm

from models.base_interviewer import InterviewerBase
from models.melu.melu import MeLU
from shared.user import WarmStartUser


class MeLUInterviewer(InterviewerBase, tt.nn.Module):
    def __init__(self, meta, use_cuda=False):
        InterviewerBase.__init__(self, meta, use_cuda=use_cuda)
        tt.nn.Module.__init__(self)

        self.decade_index, self.movie_index, self.category_index, self.person_index, \
            self.company_index = self._get_indices()

        self.entity_metadata = self._create_metadata()

        self.n_entities, self.n_decade, self.n_movies, self.n_categories, self.n_persons, self.n_companies = \
            len(self.meta.entities), len(self.decade_index), len(self.movie_index), len(self.category_index), \
            len(self.person_index), len(self.company_index)

        # Initialise variables
        self.candidate_items = None
        self.optimal_params = None
        self.keep_weight = None
        self.weight_name = None
        self.weight_len = None
        self.fast_weights = None
        self.local_lr = None
        self.global_lr = None
        self.latent = None
        self.hidden = None

        self.model = None

        if self.use_cuda:
            self.cuda()

        self.support = {}

        self.meta_optim = None
        self.local_update_target_weight_name = ['fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias',
                                                'linear_out.weight', 'linear_out.bias']

    def store_parameters(self):
        self.keep_weight = deepcopy(self.model.state_dict())
        self.weight_name = list(self.keep_weight.keys())
        self.weight_len = len(self.keep_weight)
        self.fast_weights = OrderedDict()

    def _get_indices(self):
        idx = self.meta.uri_idx
        decades = set()
        movies = set()
        categories = set()
        persons = set()
        companies = set()

        for entity, info in self.meta.entities.items():
            for label in info['labels']:
                if label == 'Decade':
                    decades.add(idx[entity])
                elif label == 'Movie':
                    movies.add(idx[entity])
                elif label == 'Category':
                    categories.add(idx[entity])
                elif label == 'Person':
                    persons.add(idx[entity])
                elif label == 'Company':
                    companies.add(idx[entity])

        decade_index = {e_idx: i for i, e_idx in enumerate(decades)}
        movie_index = {e_idx: i for i, e_idx in enumerate(movies)}
        category_index = {e_idx: i for i, e_idx in enumerate(categories)}
        person_index = {e_idx: i for i, e_idx in enumerate(persons)}
        company_index = {e_idx: i for i, e_idx in enumerate(companies)}

        return decade_index, movie_index, category_index, person_index, company_index

    def _to_onehot(self, entities, decade, movie, category, person, company):
        entity = self._create_onehot(entities, (1, self.n_entities))
        decade = self._create_onehot(decade, (1, self.n_decade))
        movie = self._create_onehot(movie, (1, self.n_movies))
        category = self._create_onehot(category, (1, self.n_categories))
        person = self._create_onehot(person, (1, self.n_persons))
        company = self._create_onehot(company, (1, self.n_companies))
        return tt.cat((entity, decade, movie, category, person, company), 1)

    def _create_onehot(self, indices, shape, multi_value=False):
        t = tt.zeros(shape)
        if len(indices) > 0:
            if multi_value:
                t[indices[:2].tolist()] = indices[-1]
            else:
                t[indices.tolist()] = 1
        return t

    def _create_metadata(self):
        triples = self.meta.triples
        e_idx_map = self.meta.uri_idx

        entities = {v: {'d': set(), 'm': set(), 'cat': set(), 'p': set(), 'com': set()}
                    for v in e_idx_map.values()}

        def append_entities(e, h, t):

            if t in self.decade_index:
                e[h]['d'].add(self.decade_index[t])
            if t in self.movie_index:
                e[h]['m'].add(self.movie_index[t])
            if t in self.category_index:
                e[h]['cat'].add(self.category_index[t])
            if t in self.person_index:
                e[h]['p'].add(self.person_index[t])
            if t in self.company_index:
                e[h]['com'].add(self.company_index[t])

        for triple in triples:
            h = triple.head
            t = triple.tail
            if h not in e_idx_map or t not in e_idx_map:
                continue

            h = e_idx_map[h]
            t = e_idx_map[t]

            append_entities(entities, h, t)
            append_entities(entities, t, h)

        entity_metadata = {e: [tt.tensor(list(metadata)) for _, metadata in m.items()] for e, m in entities.items()}

        return entity_metadata

    def _create_combined_onehots(self, entity_ids, targets=None, support=None):
        # Create data
        x = None
        for e_id in entity_ids:
            meta = self.entity_metadata[e_id]
            meta = [tt.tensor([[0, x] for x in type]).t() for type in meta]

            if support:
                e = tt.tensor([[0, item] for item, _ in support]).t()
                onehots = self._to_onehot(e, *meta)
            else:
                onehots = self._to_onehot(tt.tensor([[0, e_id]]).t(), *meta)

            if x is None:
                x = onehots
            else:
                x = tt.cat((x, onehots), 0)
        if targets:
            y = tt.tensor(targets)
            y = y.float()
        else:
            y = None

        return x, y

    def _calculate_grad_norms(self, train_data, items):
        logger.debug('Find candidate set')
        grad_norms = {}
        start = 0
        for i, (support_x, support_y, _, _) in enumerate(train_data):
            entity_vec = support_x[:, :self.n_entities]
            stop = start + len(entity_vec)
            norm = self._get_weight_avg_norm(support_x, support_y)

            for item in items[start:stop]:
                try:
                    grad_norms[item]['discriminative_value'] += norm.item()
                    grad_norms[item]['popularity_value'] += 1
                except:
                    grad_norms[item] = {
                        'discriminative_value': norm.item(),
                        'popularity_value': 1
                    }

            start = stop

        d_value_max = 0
        p_value_max = 0
        for item_id in grad_norms.keys():
            grad_norms[item_id]['discriminative_value'] /= grad_norms[item_id]['popularity_value']
            if grad_norms[item_id]['discriminative_value'] > d_value_max:
                d_value_max = grad_norms[item_id]['discriminative_value']
            if grad_norms[item_id]['popularity_value'] > p_value_max:
                p_value_max = grad_norms[item_id]['popularity_value']
        for item_id in grad_norms.keys():
            grad_norms[item_id]['discriminative_value'] /= float(d_value_max)
            grad_norms[item_id]['popularity_value'] /= float(p_value_max)
            grad_norms[item_id]['final_score'] = grad_norms[item_id]['discriminative_value'] * grad_norms[item_id][
                'popularity_value']

        return grad_norms

    def _get_all_parameters(self):
        learning_rates = [(5e-4, 5e-5)]  # [(5e-2, 5e-3), (5e-4, 5e-5), (5e-5, 5e-6), (5e-6, 5e-7)]
        latent_factors = [64, 128]  # [8, 16, 32, 64]
        hidden_units = [64, 128]  # [32, 64]
        all_params = []
        param = {}
        for learning_rate in learning_rates:
            param['lr'] = learning_rate
            for latent_factor in latent_factors:
                param['lf'] = latent_factor
                for hidden_unit in hidden_units:
                    param['hu'] = hidden_unit
                    all_params.append(param.copy())

        return all_params

    def _train(self, train_data, validation_data, batch_size, max_iteration=100, validation_limit=100):
        validation_limit = min(validation_limit, len(validation_data))
        n_batches = (len(train_data) // batch_size) + 1
        best_hitrate = -1
        best_loss = sys.float_info.max
        best_model = None
        no_increase = 0
        # Go through all epochs
        for j in range(max_iteration):
            if j == max_iteration - 1:
                logger.debug(f'Reached final iteration')
            # logger.debug(f'Starting epoch {i + 1}')
            # Ensure random order
            shuffle(train_data)

            self.model.train()
            # go through all batches
            for batch_n in range(n_batches):
                batch = train_data[batch_size * batch_n:batch_size * (batch_n + 1)]
                batch = [list(b) for b in zip(*batch)]
                self._global_update(*batch)

            # logger.debug('Starting validation')
            t = tt.ones(validation_limit)
            p = tt.zeros(validation_limit).float()
            shuffle(validation_data)
            hit = 0.
            for i, (rank, val_data, (support_x, support_y)) in enumerate(validation_data[:validation_limit]):
                lst = np.arange(len(val_data))
                if self.use_cuda:
                    preds = self._forward(support_x.cuda(), support_y.cuda(), val_data.cuda())
                else:
                    preds = self._forward(support_x, support_y, val_data)

                # Ensure that memory does not explode
                with tt.no_grad():
                    p[i] = preds[rank]
                    ordered = sorted(zip(preds, lst), reverse=True)
                    hit += 1. if rank in [r for _, r in ordered][:10] else 0.

            with tt.no_grad():
                hitrate = hit / float(validation_limit)
                loss = float(F.mse_loss(p, t))

            # Stop if no increase last two iterations.
            if hitrate <= best_hitrate:
                if hitrate == best_hitrate and loss < best_loss:
                    best_model = deepcopy(self.model.state_dict())
                if no_increase > 5:
                    break
                no_increase += 1
            else:
                best_hitrate = hitrate
                best_loss = loss
                best_model = deepcopy(self.model.state_dict())
                no_increase = 0

        return best_hitrate, best_model

    def _forward(self, support_set_x, support_set_y, query_set_x, num_local_update=1):
        for idx in range(num_local_update):
            if idx > 0:
                self.model.load_state_dict(self.fast_weights)
            weight_for_local_update = list(self.model.state_dict().values())
            support_set_y_pred = self.model(support_set_x)
            loss = F.mse_loss(support_set_y_pred, support_set_y.view(-1, 1))
            self.model.zero_grad()
            grad = tt.autograd.grad(loss, self.model.parameters(), create_graph=True)

            # local update
            for i in range(self.weight_len):
                if self.weight_name[i] in self.local_update_target_weight_name:
                    self.fast_weights[self.weight_name[i]] = weight_for_local_update[i] - self.local_lr * grad[i]
                else:
                    self.fast_weights[self.weight_name[i]] = weight_for_local_update[i]
        self.model.load_state_dict(self.fast_weights)
        query_set_y_pred = self.model(query_set_x)
        self.model.load_state_dict(self.keep_weight)
        return query_set_y_pred

    def _global_update(self, support_set_xs, support_set_ys, query_set_xs, query_set_ys, num_local_update=1):
        batch_sz = len(support_set_xs)
        losses_q = []
        if self.use_cuda:
            for i in range(batch_sz):
                support_set_xs[i] = support_set_xs[i].cuda()
                support_set_ys[i] = support_set_ys[i].cuda()
                query_set_xs[i] = query_set_xs[i].cuda()
                query_set_ys[i] = query_set_ys[i].cuda()
        for i in range(batch_sz):
            query_set_y_pred = self._forward(support_set_xs[i], support_set_ys[i], query_set_xs[i], num_local_update)
            loss_q = F.mse_loss(query_set_y_pred, query_set_ys[i].view(-1, 1))
            losses_q.append(loss_q)
        losses_q = tt.stack(losses_q).mean(0)
        self.meta_optim.zero_grad()
        losses_q.backward()
        self.meta_optim.step()
        self.store_parameters()
        return

    def _get_weight_avg_norm(self, support_set_x, support_set_y, num_local_update=1):
        tmp = 0.
        if self.use_cuda:
            support_set_x = support_set_x.cuda()
            support_set_y = support_set_y.cuda()
        for idx in range(num_local_update):
            if idx > 0:
                self.model.load_state_dict(self.fast_weights)
            weight_for_local_update = list(self.model.state_dict().values())
            support_set_y_pred = self.model(support_set_x)
            loss = F.mse_loss(support_set_y_pred, support_set_y.view(-1, 1))
            # unit loss
            loss /= tt.norm(loss).tolist()
            self.model.zero_grad()
            grad = tt.autograd.grad(loss, self.model.parameters(), create_graph=True)
            for i in range(self.weight_len):
                # For averaging Forbenius norm.
                tmp += tt.norm(grad[i])
                if self.weight_name[i] in self.local_update_target_weight_name:
                    self.fast_weights[self.weight_name[i]] = weight_for_local_update[i] - self.local_lr * grad[i]
                else:
                    self.fast_weights[self.weight_name[i]] = weight_for_local_update[i]
        return tmp / num_local_update

    def warmup(self, training: Dict[int, WarmStartUser], interview_length=5):
        user_ratings = {}
        items = []
        for user, datasets in training.items():
            ratings = list(datasets.training.items())
            tmp = max(len(ratings) - 3, len(ratings) // 2)
            num_support = min(tmp, 10)
            shuffle(ratings)
            support = ratings[:num_support]
            query = ratings[num_support:]
            if not support or not query:
                continue
            user_ratings[user] = [support, query]
            items.extend([e_id for e_id, _ in support])

        del user, datasets, ratings, tmp, num_support, support, query

        support_xs = []
        support_ys = []
        query_xs = []
        query_ys = []

        logger.debug(f'Creating train data')
        for user, (support, query) in user_ratings.items():
            support_x, support_y = self._create_combined_onehots(*zip(*support))
            query_x, query_y = self._create_combined_onehots(*zip(*query), support)

            support_xs.append(support_x)
            support_ys.append(support_y)

            tmp = [support_x, support_y]  # [tt.cat((support_x, query_x),0), tt.cat((support_y, query_y), 0)]
            user_ratings[user].append(tmp)
            self.support[user] = [support, tmp]

            query_xs.append(query_x)
            query_ys.append(query_y)

        train_data = list(zip(support_xs, support_ys, query_xs, query_ys))
        validation = list(training.items())
        del support_xs, support_ys, query_xs, query_ys, support_x, support_y, query_x, query_y, tmp, user, support, \
            query, training

        val = []
        logger.debug(f'Creating validation set')
        for user, warm in validation[:10]:
            if user not in user_ratings:
                continue
            pos_sample = warm.validation['positive']
            neg_samples = warm.validation['negative']
            support, query, support_train = user_ratings[user]
            samples = np.array([pos_sample] + neg_samples)
            shuffle(samples)
            rank = np.argwhere(samples == pos_sample)[0]

            u_val, _ = self._create_combined_onehots(samples)

            val.append((rank, u_val, support_train))

        del user_ratings, validation

        batch_size = 32

        logger.debug('Starting training')
        if self.optimal_params is None:
            best_param = None
            best_hitrate = -1
            best_model = None
            for param in self._get_all_parameters():
                logger.debug(f'Trying with params: {param}')
                self.load_parameters(param)
                hr, model = self._train(train_data, val, batch_size)
                if hr > best_hitrate:
                    logger.debug(f'New best param with HR:{hr}')
                    best_hitrate = hr
                    best_model = model
                    best_param = param.copy()

            self.load_parameters(best_param)
            self.model.load_state_dict(best_model)
            self.store_parameters()
            logger.debug(f'Found best param with HR:{best_hitrate}')
        else:
            self.load_parameters(self.optimal_params)
            _, _ = self._train(train_data, val, batch_size)

        grad_norms = self._calculate_grad_norms(train_data, items)

        self.candidate_items = list(sorted(grad_norms.keys(), key=lambda x: grad_norms[x]['final_score'],
                                           reverse=True))

    def predict(self, items: List[int], answers: Dict) -> Dict[int, float]:
        support_x, support_y = self._create_combined_onehots(*zip(*answers.items()))
        query, _ = self._create_combined_onehots(items, support=answers.items())

        if self.use_cuda:
            support_x, support_y, query = support_x.cuda(), support_y.cuda(), query.cuda()

        preds = self._forward(support_x, support_y, query)
        return {k: v for k, v in zip(items, preds)}

    def interview(self, answers: Dict, max_n_questions=5) -> List[int]:
        return self.candidate_items[:max_n_questions]

    def get_parameters(self):
        return self.optimal_params

    def load_parameters(self, params):
        self.optimal_params = params
        self.local_lr, self.global_lr = params['lr']
        self.latent = params['lf']
        self.hidden = params['hu']

        self.model = MeLU(self.n_entities, self.n_decade, self.n_movies, self.n_categories, self.n_persons,
             self.n_companies, self.latent, self.hidden)

        if self.use_cuda:
            self.model.cuda()
            if tt.cuda.device_count() > 1:
                self.model = tt.nn.DataParallel(self.model)

        self.store_parameters()

        self.meta_optim = tt.optim.Adam(self.model.parameters(), lr=self.global_lr)
