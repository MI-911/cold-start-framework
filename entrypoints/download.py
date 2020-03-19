import argparse
import sys
import os

from loguru import logger

from shared.validators import valid_dir
from urllib.request import urlretrieve

BASE_URL = 'https://mindreader.tech/api'

parser = argparse.ArgumentParser()
parser.add_argument('--output', nargs=1, type=valid_dir, help='path to store downloaded data')


def download_entities(path):
    logger.info('Downloading entities')

    urlretrieve(f'{BASE_URL}/entities', os.path.join(path, 'entities.csv'))


def download_triples(path):
    logger.info('Downloading triples')

    urlretrieve(f'{BASE_URL}/triples', os.path.join(path, 'triples.csv'))


def download_ratings(path):
    logger.info('Downloading ratings')

    urlretrieve(f'{BASE_URL}/ratings?versions=100k,100k-newer,100k-fix', os.path.join(path, 'ratings.csv'))


def download_triples(path):
    logger.info('Downloading triples')

    urlretrieve(f'{BASE_URL}/triples', os.path.join(path, 'triples.csv'))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    output_path = args.output[0]

    download_ratings(output_path)
    download_entities(output_path)
    download_triples(output_path)
