import argparse
import sys
import os

from loguru import logger

from partitioners import partition_interview
from shared.validators import valid_dir

parser = argparse.ArgumentParser()
parser.add_argument('--input', nargs=1, type=valid_dir, help='path to sources/input data')
parser.add_argument('--output', nargs=1, type=valid_dir, help='path to output data')

if __name__ == '__main__':
    logger.info(f'Working directory: {os.getcwd()}')

    if len(sys.argv) < 3:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    partition_interview.partition(input_directory=args.input[0], output_directory=args.output[0])
