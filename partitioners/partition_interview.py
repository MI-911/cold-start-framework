import os
import pickle
from math import ceil
from typing import List, Dict, Set

import numpy as np
import pandas as pd
import tqdm
from loguru import logger
from pandas import DataFrame

from experiments.experiment import ExperimentOptions, CountFilter, Sentiment, sentiment_to_int, EntityType, \
    RankingOptions
from shared.enums import UnseenSampling, SeenSampling
from shared.graph_triple import GraphTriple
from shared.meta import Meta
from shared.ranking import Ranking
from shared.user import WarmStartUser, ColdStartUserSet, ColdStartUser


def _sample_seen(ratings: DataFrame, sentiment: Sentiment, remove_items: List = None, n_items=1):
    items = sorted(ratings[(ratings.sentiment == sentiment_to_int(sentiment)) & ratings.isItem].entityIdx.unique())

    if remove_items:
        items = [item for item in items if item not in remove_items]

    if not items:
        return []

    return np.random.choice(items, size=n_items, replace=False)


def _choice(lst, count, probabilities):
    return np.random.choice(lst, size=count, replace=False, p=np.array(probabilities) / np.sum(probabilities))


def _get_unseen_weights(item_ratings, unseen_items: List, options: RankingOptions, positive_items: List[int] = None,
                        alpha=10):
    if options.unseen_sampling == UnseenSampling.RANDOM:
        return [1 for _ in unseen_items]

    entity_weight = dict(zip(item_ratings['entityIdx'], item_ratings['num_ratings']))

    if options.unseen_sampling == UnseenSampling.EQUAL_POPULARITY and positive_items:
        positive_ratings = np.mean([entity_weight[item] for item in positive_items])

        entity_weight = {e: pow(abs(w - positive_ratings) + 1, -alpha) for e, w in entity_weight.items()}

    return [entity_weight[entity] for entity in unseen_items]


def _sample_unseen(ratings: DataFrame, user_id: int, options: RankingOptions, positive_items: List[int] = None):
    item_ratings = ratings[ratings.isItem]

    seen_items = set(item_ratings[item_ratings.userId == user_id].entityIdx.unique())
    unseen_items = sorted(set(item_ratings.entityIdx.unique()).difference(seen_items))

    unseen_weights = _get_unseen_weights(item_ratings, unseen_items, options, positive_items)

    return _choice(unseen_items, options.sentiment_count.get(Sentiment.UNSEEN, 0), unseen_weights)


def _get_ratings_dict(from_ratings):
    return {row.entityIdx: row.sentiment for _, row in from_ratings.iterrows()}


def _chunks(lst, ratio):
    return [set(arr) for arr in np.array_split(lst, int(1 / ratio))]


def _get_ratings(ratings_path, include_unknown, cold_start_ratio, count_filters: List[CountFilter]):
    ratings = pd.read_csv(ratings_path)
    if not include_unknown:
        ratings = ratings[ratings.sentiment != 0]

    # Find duplicate (user, item) pairs
    duplicates = ratings.groupby(['uri', 'userId']).size()
    if len(duplicates[duplicates > 1]):
        logger.error('Found duplicate (user, uri) pairs')

        exit(1)

    # Compute ratings per entity
    entity_ratings = ratings[['uri', 'userId']].groupby('uri').count()
    entity_ratings.columns = ['num_ratings']

    # Add number of ratings to ratings, used for entity sampling
    ratings = ratings.merge(entity_ratings, on='uri')

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
    users = sorted(ratings['userId'].unique())
    np.random.shuffle(users)

    splits = list()

    # Create splits corresponding to 1 / cold_start_ratio
    for cold_start_users in _chunks(users, cold_start_ratio):
        warm_start_users = set(users) - set(cold_start_users)

        assert warm_start_users.isdisjoint(cold_start_users)
        assert len(cold_start_users) + len(warm_start_users) == len(users)

        splits.append((warm_start_users, cold_start_users))

    # Assert that no cold start user appears twice
    for i, (_, cold_users) in enumerate(splits):
        for j, (_, other_cold_users) in enumerate(splits):
            if i == j:
                continue

            assert not cold_users.intersection(other_cold_users)

    return ratings, users, splits


def _get_training_data(experiment: ExperimentOptions, ratings, warm_start_users, user_idx,
                       short_head_items) -> Dict[int, WarmStartUser]:
    training_data = dict()

    progress = tqdm.tqdm(sorted(warm_start_users))
    for user in progress:
        progress.set_description(f'Processing warm-start user {user}')

        u_ratings, ranking = _get_ranking(ratings, user, experiment.ranking_options, short_head_items)

        # Verify that all required samples are present
        if len(set(ranking.to_list())) != experiment.ranking_options.get_num_total():
            logger.warning(f'Could not sample all required entities for warm-start user {user}, skipping')

            continue

        training_dict = _get_ratings_dict(u_ratings)

        # Assert ranking samples not in training
        assert not set(ranking.to_list()).intersection(training_dict.keys())

        training_data[user_idx[user]] = WarmStartUser(training_dict, ranking)

    return training_data


def _get_testing_data(experiment: ExperimentOptions, ratings, cold_start_users, user_idx, movie_indices,
                      short_head_items) -> Dict[int, ColdStartUser]:
    testing_data = dict()

    progress = tqdm.tqdm(sorted(cold_start_users))
    for user in progress:
        progress.set_description(f'Processing cold-start user {user}')

        # For each positive item, create an answer set with that item left out
        sets = []
        for _ in range(experiment.evaluation_samples):
            u_ratings, ranking = _get_ranking(ratings, user, experiment.ranking_options, short_head_items)
            answer_dict = _get_ratings_dict(u_ratings)

            # Verify that all required samples are present
            if len(set(ranking.to_list())) != experiment.ranking_options.get_num_total():
                logger.warning(f'Could not sample all required entities for cold-start user {user}, skipping')

                continue

            # Skip if user cannot provide any movie answers
            # By checking this, we can remove DEs from answer sets without losing users between comparisons
            if not set(answer_dict.keys()).intersection(movie_indices):
                logger.warning(f'No item answers for cold-start user {user}, skipping')

                continue

            sets.append(ColdStartUserSet(answer_dict, ranking))

        testing_data[user_idx[user]] = ColdStartUser(sets)

    return testing_data


def _get_ranking(ratings: DataFrame, user_id: int, options: RankingOptions, short_head_items) -> (DataFrame, Ranking):
    u_ratings = ratings[ratings.userId == user_id]

    # Create ranking instance holding all samples to rank
    ranking = Ranking()
    for sentiment in sorted(options.sentiment_count.keys()):
        if sentiment == Sentiment.UNSEEN:
            continue

        # If sampling the long tail, do not sample items from the short head
        remove_items = short_head_items if options.seen_sampling == SeenSampling.LONG_TAIL else None

        ranking.sentiment_samples[sentiment] = _sample_seen(u_ratings, sentiment, remove_items,
                                                            options.sentiment_count.get(sentiment, 0))

    # Handle unseen items separately, as it is based on the popularity of the sampled seen items
    ranking.sentiment_samples[Sentiment.UNSEEN] = _sample_unseen(ratings, user_id, options, ranking.get_seen_samples())

    # Return user's ratings without items to rank
    return u_ratings[~u_ratings.entityIdx.isin(ranking.to_list())], ranking


def _get_entities(entities_path):
    entities = pd.read_csv(entities_path)

    return {row['uri']: {'name': row['name'], 'labels': row['labels'].split('|')} for _, row in entities.iterrows()}


def _load_triples(triples_path):
    triples = pd.read_csv(triples_path)

    return [GraphTriple(row['head_uri'], row['relation'], row['tail_uri']) for _, row in triples.iterrows()]


def partition(experiment: ExperimentOptions, input_directory, output_directory):
    ratings_path = os.path.join(input_directory, experiment.ratings_file)
    entities_path = os.path.join(input_directory, 'entities.csv')
    triples_path = os.path.join(input_directory, 'triples.csv')

    entities = _get_entities(entities_path)
    np.random.seed(experiment.seed)

    # Load ratings data
    ratings, users, splits = _get_ratings(ratings_path, experiment.include_unknown, experiment.cold_start_ratio,
                                          count_filters=experiment.count_filters)

    for idx, (warm_users, cold_users) in enumerate(splits):
        split_name = f'split_{idx}'
        split_output_directory = os.path.join(os.path.join(output_directory, experiment.name), split_name)

        logger.info(f'Creating split {experiment.name}/{split_name}')

        _create_split(experiment, entities, split_output_directory, triples_path, ratings, users, warm_users,
                      cold_users)


def _get_rated_entities(training_data: Dict[int, WarmStartUser]):
    rated_entities = set()

    for _, ratings in training_data.items():
        for entity, rating in ratings.training.items():
            rated_entities.add(entity)

    return rated_entities


def _get_short_head_items(ratings_df: DataFrame):
    item_ratings = dict(zip(ratings_df['entityIdx'], ratings_df['num_ratings']))
    popular_items = [entity for entity, ratings in sorted(item_ratings.items(), key=lambda pair: pair[1], reverse=True)]

    return popular_items[:ceil(0.02 * len(popular_items))]


def _create_split(experiment: ExperimentOptions, entities, output_directory: str, triples_path: str, ratings, users,
                  warm_users, cold_users):
    # Optionally limit entities to available URIs
    if experiment.limit_entities:
        uris = set(ratings.uri.unique())
        entities = {entity: _ for entity, _ in entities.items() if entity in uris}

    # Map users and entities to indices
    user_idx = {k: v for v, k in enumerate(sorted(users))}
    entity_idx = {k: v for v, k in enumerate(sorted(entities.keys()))}
    ratings['entityIdx'] = ratings.uri.transform(entity_idx.get)

    # Find movie indices
    movie_ratings = ratings[ratings.isItem]
    movie_indices = set(movie_ratings.entityIdx.unique())

    # Find items in the long tail if specified
    short_head_items = _get_short_head_items(movie_ratings)
    logger.debug(f'Short head items: {len(short_head_items)}/{len(movie_indices)}')

    # Partition training data from users
    training_data = _get_training_data(experiment, ratings, warm_users, user_idx, short_head_items)

    # Partition testing data, limit to observed entities
    rated_entities = _get_rated_entities(training_data)
    testing_data = _get_testing_data(experiment, ratings[ratings.entityIdx.isin(rated_entities)], cold_users, user_idx,
                                     movie_indices, short_head_items)

    logger.info(f'Created {len(training_data)} training entries and {len(testing_data)} testing entries')

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
                         default_cutoff=experiment.ranking_options.default_cutoff,
                         sentiment_utility=experiment.ranking_options.sentiment_utility,
                         validator=experiment.validator), fp)
