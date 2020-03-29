import argparse
import os
import sys

from loguru import logger

from experiments.experiment import ExperimentOptions, CountFilter, Sentiment, EntityType
from partitioners import partition_interview
from shared.validators import valid_dir

parser = argparse.ArgumentParser()
parser.add_argument('--input', nargs=1, type=valid_dir, help='path to sources/input data')
parser.add_argument('--output', nargs=1, type=valid_dir, help='path to output data')

experiments = [
    ExperimentOptions(name='default', split_seeds=[42, 120], count_filters=[
        CountFilter(lambda count: count >= 2, entity_type=EntityType.RECOMMENDABLE, sentiment=Sentiment.POSITIVE),
        CountFilter(lambda count: count >= 1, entity_type=EntityType.DESCRIPTIVE, sentiment=Sentiment.ANY)
    ])
]

if __name__ == '__main__':
    logger.info(f'Working directory: {os.getcwd()}')

    if len(sys.argv) < 3:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    for experiment in experiments:
        partition_interview.partition(experiment, input_directory=args.input[0], output_directory=args.output[0])
