import argparse
import sys

from partitioners.partition_interview import partition

parser = argparse.ArgumentParser()
parser.add_argument('--input', nargs=1, type=str, help='path to sources/input data')
parser.add_argument('--output', nargs=1, type=str, help='path to output data')

if __name__ == '__main__':
    if len(sys.argv) < 3:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    partition(input_directory=args.input, output_directory=args.output)
