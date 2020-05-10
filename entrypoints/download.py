import argparse
import sys
import os

import pandas as pd
from loguru import logger

from shared.utility import valid_dir
from urllib.request import urlretrieve

BASE_URL = 'https://mindreader.tech/api'

parser = argparse.ArgumentParser()
parser.add_argument('--output', nargs=1, type=valid_dir, help='path to store downloaded data')


def download_entities(path):
    logger.info('Downloading entities')

    entities = os.path.join(path, 'entities.csv')
    urlretrieve(f'{BASE_URL}/entities', entities)

    return entities


def download_triples(path):
    logger.info('Downloading triples')

    triples = os.path.join(path, 'triples.csv')
    urlretrieve(f'{BASE_URL}/triples', triples)

    return triples


def download_ratings(path):
    logger.info('Downloading ratings')

    ratings = os.path.join(path, 'ratings.csv')
    urlretrieve(f'{BASE_URL}/ratings?versions=100k,100k-newer,100k-fix&final=yes', ratings)

    return ratings


if __name__ == '__main__':
    if len(sys.argv) < 2:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    output_path = args.output[0]

    entities_df = pd.read_csv(download_entities(output_path))
    entity_uris = set(entities_df.uri.unique())

    ratings_file = download_ratings(output_path)
    ratings_df = pd.read_csv(ratings_file)
    ratings_df = ratings_df[ratings_df.uri.isin(entity_uris)]
    with open(ratings_file, 'w') as fp:
        fp.write(ratings_df.to_csv(index=True))

    download_triples(output_path)
