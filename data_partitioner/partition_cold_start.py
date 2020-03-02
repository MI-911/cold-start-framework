import pandas as pd
import random
import json
import numpy as np
import tqdm
from loguru import logger
from os import path

from models.shared.user import WarmStartUser, ColdStartUser, ColdStartUserSet


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, WarmStartUser):
            return {'validation': obj.validation, 'training': obj.training}
        elif isinstance(obj, ColdStartUser):
            return {'validation': obj.validation, 'sets': obj.sets}
        elif isinstance(obj, ColdStartUserSet):
            return {'positive': obj.positive, 'negative': obj.negative, 'answers': obj.answers}
        else:
            return super(NpEncoder, self).default(obj)


def _sample_positive(from_ratings):
    return random.choice(from_ratings[from_ratings.sentiment == 1 & from_ratings.isItem].entityIdx.unique())


def _sample_unseen_items(ratings, user_id, n_items=100):
    item_ratings = ratings[ratings.isItem]

    seen_items = set(item_ratings[item_ratings.userId == user_id].entityIdx.unique())
    unseen_items = list(set(item_ratings.entityIdx.unique()).difference(seen_items))

    random.shuffle(unseen_items)

    return unseen_items[:n_items]


def _get_ratings_dict(from_ratings):
    return {row.entityIdx: row.sentiment for _, row in from_ratings.iterrows()}


def _get_validation_dict(ratings, user_id, left_out):
    return {
        'positive': left_out,
        'negative': _sample_unseen_items(ratings, user_id)
    }


def _get_ratings(ratings_path, include_unknown, warm_start_ratio):
    ratings = pd.read_csv(ratings_path)
    if not include_unknown:
        ratings = ratings[ratings.sentiment != 0]

    # Compute ratings per entity
    # In the future, this could be used for popularity sampling of negative samples
    entity_ratings = ratings[['uri', 'userId']].groupby('uri').count()
    entity_ratings.columns = ['num_ratings']

    # Filter users with less than two positive movie samples
    tmp = ratings[ratings.sentiment == 1 & ratings.isItem][['uri', 'userId']].groupby('userId').count()
    tmp.columns = ['pos_ratings']

    ratings = ratings[ratings.userId.isin(tmp[tmp.pos_ratings >= 2].index)]

    # Partition into warm and cold start users
    users = ratings['userId'].unique()
    random.shuffle(users)

    num_warm_start = int(len(users) * warm_start_ratio)
    warm_start_users = set(users[:num_warm_start])
    cold_start_users = set(users[num_warm_start:])

    assert warm_start_users.isdisjoint(cold_start_users)

    return ratings, warm_start_users, cold_start_users, users


def _get_training_data(ratings, warm_start_users, user_idx):
    training_data = dict()

    for user in tqdm.tqdm(warm_start_users):
        u_ratings = ratings[ratings.userId == user]

        val_sample = _sample_positive(u_ratings)

        training_dict = _get_ratings_dict(u_ratings[u_ratings.entityIdx != val_sample])
        validation_dict = _get_validation_dict(ratings, user, val_sample)

        # Assert validation sample not in training
        assert val_sample not in training_dict.keys()

        # Assert positive sample not in negative samples
        assert val_sample not in validation_dict['negative']

        # Assert negative samples not in training
        assert not set(validation_dict['negative']).intersection(training_dict.keys())

        training_data[user_idx[user]] = WarmStartUser(training_dict, validation_dict)

    return training_data


def _get_testing_data(ratings, cold_start_users, user_idx, movie_indices):
    testing_data = dict()

    for user in tqdm.tqdm(cold_start_users):
        u_ratings = ratings[ratings.userId == user]

        # Before exhaustive LOO, get validation sample
        val_sample = _sample_positive(u_ratings)
        validation_dict = _get_validation_dict(ratings, user, val_sample)

        # For convenience, leave out the validation sample from the user's ratings
        u_ratings = u_ratings[u_ratings.entityIdx != val_sample]

        # Find all the user's positive item ratings
        u_pos = u_ratings[u_ratings.isItem & u_ratings.sentiment == 1]
        assert len(u_pos)

        # For each positive item, create an answer set with that item left out
        sets = []
        for idx, pos in u_pos.iterrows():
            answer_dict = _get_ratings_dict(u_ratings[u_ratings.entityIdx != pos.entityIdx])
            pos_neg_dict = _get_validation_dict(ratings, user, pos.entityIdx)

            # Skip if user cannot provide any movie answers
            # By checking this, we can remove DEs from answer sets without losing users between comparisons
            if not set(answer_dict.keys()).intersection(movie_indices):
                continue

            # Assert that the positive item is not in the negative samples
            assert pos.entityIdx not in pos_neg_dict['negative']

            # Assert that user cannot answer about the positive item
            assert pos.entityIdx not in answer_dict

            sets.append(ColdStartUserSet(answer_dict, **pos_neg_dict))

        # Check if user has any valid answer sets
        if not sets:
            continue

        testing_data[user_idx[user]] = ColdStartUser(sets, validation_dict)

    return testing_data


def _get_entities(entities_path):
    entities = pd.read_csv(entities_path)

    return {row['uri']: {'name': row['name'], 'labels': row['labels'].split('|')} for idx, row in entities.iterrows()}


def partition(ratings_path, entities_path, output_directory, random_seed=42, warm_start_ratio=0.75,
              include_unknown=False):
    random.seed(random_seed)

    # Load ratings data
    ratings, warm_users, cold_users, users = _get_ratings(ratings_path, include_unknown, warm_start_ratio)

    # Map users and entities to indices
    user_idx = {k: v for v, k in enumerate(users)}
    entity_idx = {k: v for v, k in enumerate(ratings['uri'].unique())}

    ratings['entityIdx'] = ratings.uri.transform(entity_idx.get)

    # Find movie indices
    movie_indices = set(ratings[ratings.isItem].entityIdx.unique())

    # Partition training/testing data from users
    training_data = _get_training_data(ratings, warm_users, user_idx)
    testing_data = _get_testing_data(ratings, cold_users, user_idx, movie_indices)

    logger.info(f'Created {len(training_data)} training entries and {len(testing_data)} testing entries')

    # Write data to disk
    logger.info(f'Writing dataset to {output_directory}')
    with open(path.join(output_directory, 'training.json'), 'w') as fp:
        json.dump(training_data, fp, cls=NpEncoder)

    with open(path.join(output_directory, 'testing.json'), 'w') as fp:
        json.dump(testing_data, fp, cls=NpEncoder)

    with open(path.join(output_directory, 'meta.json'), 'w') as fp:
        json.dump({
            'entities': _get_entities(entities_path),
            'uri_idx': entity_idx,
            'idx_item': {row.entityIdx: row.isItem for idx, row in ratings.iterrows()}
        }, fp, cls=NpEncoder)


if __name__ == '__main__':
    partition(ratings_path='../sources/mindreader/ratings-100k.csv', entities_path='../sources/mindreader/entities.csv',
              output_directory='data')
