import os
import pickle
import random
from typing import List

import pandas as pd
import tqdm
from loguru import logger
from pandas import DataFrame

from experiments.experiment import ExperimentOptions, CountFilter, Sentiment, sentiment_to_int, EntityType, \
    RankingOptions
from shared.graph_triple import GraphTriple
from shared.meta import Meta
from shared.ranking import Ranking
from shared.user import WarmStartUser, ColdStartUserSet, ColdStartUser


def _sample_sentiment(ratings, sentiment: Sentiment, n_items=1):
    items = list(ratings[ratings.sentiment == sentiment_to_int(sentiment) & ratings.isItem].entityIdx.unique())
    random.shuffle(items)

    return items[:n_items]


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


def _get_ratings(ratings_path, include_unknown, warm_start_ratio, count_filters: List[CountFilter]):
    ratings = pd.read_csv(ratings_path)
    if not include_unknown:
        ratings = ratings[ratings.sentiment != 0]

    # Compute ratings per entity
    # In the future, this could be used for popularity sampling of negative samples
    entity_ratings = ratings[['uri', 'userId']].groupby('uri').count()
    entity_ratings.columns = ['num_ratings']

    # Filter users with count filters
    if count_filters:
        # Consider the most generic filters first (i.e. any sentiment)
        count_filters = sorted(count_filters,
                               key=lambda f: int(f.sentiment != Sentiment.ANY) + int(f.entity_type != EntityType.ANY))

        for count_filter in count_filters:
            df_tmp = ratings

            # Filter entity type
            if count_filter.entity_type != EntityType.ANY:
                df_tmp = df_tmp[df_tmp.isItem == (count_filter.entity_type == EntityType.RECOMMENDABLE)]

            # Filter sentiment
            if count_filter.sentiment != Sentiment.ANY:
                df_tmp = df_tmp[df_tmp.sentiment == sentiment_to_int(count_filter.sentiment)]

            # Group ratings by user
            df_tmp = df_tmp[['uri', 'userId']].groupby('userId').count()
            df_tmp.columns = ['num_ratings']

            ratings = ratings[ratings.userId.isin(df_tmp[count_filter.filter_func(df_tmp.num_ratings)].index)]

    # Partition into warm and cold start users
    users = ratings['userId'].unique()
    random.shuffle(users)

    num_warm_start = int(len(users) * warm_start_ratio)
    warm_start_users = set(users[:num_warm_start])
    cold_start_users = set(users[num_warm_start:])

    assert warm_start_users.isdisjoint(cold_start_users)

    return ratings, warm_start_users, cold_start_users, users


def _get_training_data(experiment: ExperimentOptions, ratings, warm_start_users, user_idx):
    training_data = dict()

    progress = tqdm.tqdm(warm_start_users)
    for user in progress:
        progress.set_description(f'Processing warm-start user {user}')

        u_ratings, ranking = _get_ranking(ratings, user, experiment.ranking_options)

        training_dict = _get_ratings_dict(u_ratings)

        # Assert negative samples not in training
        assert not set(ranking.to_rank).intersection(training_dict.keys())

        training_data[user_idx[user]] = WarmStartUser(training_dict, ranking)

    return training_data


def _get_testing_data(experiment: ExperimentOptions, ratings, cold_start_users, user_idx, movie_indices):
    testing_data = dict()

    progress = tqdm.tqdm(cold_start_users)
    for user in progress:
        progress.set_description(f'Processing cold-start user {user}')

        # For each positive item, create an answer set with that item left out
        sets = []
        for _ in range(experiment.evaluation_samples):
            u_ratings, ranking = _get_ranking(ratings, user, experiment.ranking_options)
            answer_dict = _get_ratings_dict(u_ratings)

            # Skip if user cannot provide any movie answers
            # By checking this, we can remove DEs from answer sets without losing users between comparisons
            if not set(answer_dict.keys()).intersection(movie_indices):
                continue

            sets.append(ColdStartUserSet(answer_dict, ranking))

        testing_data[user_idx[user]] = ColdStartUser(sets)

    return testing_data


def _get_ranking(ratings: DataFrame, user_id: int, ranking_options: RankingOptions) -> (DataFrame, Ranking):
    u_ratings = ratings[ratings.userId == user_id]

    unseen = _sample_unseen_items(ratings, user_id, n_items=ranking_options.num_unseen)
    positive = _sample_sentiment(u_ratings, Sentiment.POSITIVE, n_items=ranking_options.num_positive)
    negative = _sample_sentiment(u_ratings, Sentiment.NEGATIVE, n_items=ranking_options.num_negative)
    unknown = _sample_sentiment(u_ratings, Sentiment.UNKNOWN, n_items=ranking_options.num_unknown)

    # Create ranking instance, containing all items to rank and a separate reference to the positives
    ranking = Ranking(unseen + positive + negative + unknown, positive)

    # Assert that we were able to sample all requested items
    assert len(ranking.to_rank) == ranking_options.get_num_total()

    # Return user's ratings without items to rank
    return u_ratings[~u_ratings.entityIdx.isin(ranking.to_rank)], ranking


def _get_entities(entities_path):
    entities = pd.read_csv(entities_path)

    return {row['uri']: {'name': row['name'], 'labels': row['labels'].split('|')} for _, row in entities.iterrows()}


def _load_triples(triples_path):
    triples = pd.read_csv(triples_path)

    return [GraphTriple(row['head_uri'], row['relation'], row['tail_uri']) for _, row in triples.iterrows()]


def partition(experiment: ExperimentOptions, input_directory, output_directory):
    ratings_path = os.path.join(input_directory, 'ratings.csv')
    entities_path = os.path.join(input_directory, 'entities.csv')
    triples_path = os.path.join(input_directory, 'triples.csv')

    entities = _get_entities(entities_path)

    for idx, seed in enumerate(experiment.split_seeds):
        split_name = f'split_{idx}'
        split_output_directory = os.path.join(os.path.join(output_directory, experiment.name), split_name)

        logger.info(f'Partitioning {experiment.name}/{split_name}')

        partition_seed(experiment, seed, entities, split_output_directory, ratings_path, triples_path)


def partition_seed(experiment: ExperimentOptions, seed: int, entities, output_directory: str,
                   ratings_path: str, triples_path: str):
    random.seed(seed)

    # Load ratings data
    ratings, warm_users, cold_users, users = _get_ratings(ratings_path, experiment.include_unknown,
                                                          experiment.warm_start_ratio,
                                                          count_filters=experiment.count_filters)

    # Map users and entities to indices
    user_idx = {k: v for v, k in enumerate(users)}
    entity_idx = {k: v for v, k in enumerate(set(entities.keys()))}
    ratings['entityIdx'] = ratings.uri.transform(entity_idx.get)

    # Find movie indices
    movie_indices = set(ratings[ratings.isItem].entityIdx.unique())

    # Partition training/testing data from users
    training_data = _get_training_data(experiment, ratings, warm_users, user_idx)
    testing_data = _get_testing_data(experiment, ratings, cold_users, user_idx, movie_indices)

    logger.info(f'Created {len(training_data)} training entries and {len(testing_data)} testing entries')

    # Write data to disk
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    logger.info(f'Writing dataset to {output_directory}')
    with open(os.path.join(output_directory, 'training.pkl'), 'wb') as fp:
        pickle.dump(training_data, fp)

    with open(os.path.join(output_directory, 'testing.pkl'), 'wb') as fp:
        pickle.dump(testing_data, fp)

    with open(os.path.join(output_directory, 'meta.pkl'), 'wb') as fp:
        pickle.dump(Meta(entities=entities, uri_idx=entity_idx, users=list(user_idx.values()),
                         idx_item={row.entityIdx: row.isItem for idx, row in ratings.iterrows()},
                         recommendable_entities=list(movie_indices), triples=_load_triples(triples_path),
                         default_cutoff=experiment.ranking_options.default_cutoff), fp)
