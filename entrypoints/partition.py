import argparse
import os
import sys

from loguru import logger

from configurations.experiments import experiments, experiment_names
from partitioners import partition_interview
from shared.utility import valid_dir

parser = argparse.ArgumentParser()
parser.add_argument('--input', nargs=1, type=valid_dir, help='path to sources/input data')
parser.add_argument('--output', nargs=1, type=valid_dir, help='path to output data')
parser.add_argument('--experiments', nargs='*', type=str, choices=experiment_names, help='path to output data')

if __name__ == '__main__':
    logger.info(f'Working directory: {os.getcwd()}')

    if len(sys.argv) < 3:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    for experiment in sorted(experiments, key=lambda e: e.name):
        if args.experiments and experiment.name not in args.experiments:
            logger.info(f'Skipping partitioning of {experiment.name}')

            continue

        partition_interview.partition(experiment, input_directory=args.input[0], output_directory=args.output[0])
