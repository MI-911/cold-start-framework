import argparse
import os
import sys

from loguru import logger

from shared.enums import EntityType, Sentiment, Metric
from experiments.experiment import ExperimentOptions, CountFilter, RankingOptions
from partitioners import partition_interview
from shared.utility import valid_dir
from shared.validator import Validator

parser = argparse.ArgumentParser()
parser.add_argument('--input', nargs=1, type=valid_dir, help='path to sources/input data')
parser.add_argument('--output', nargs=1, type=valid_dir, help='path to output data')

separation = ExperimentOptions(name='separation', split_seeds=[42], count_filters=[
        CountFilter(lambda count: count >= 1, entity_type=EntityType.RECOMMENDABLE, sentiment=Sentiment.POSITIVE),
        CountFilter(lambda count: count >= 1, entity_type=EntityType.RECOMMENDABLE, sentiment=Sentiment.NEGATIVE),
        CountFilter(lambda count: count >= 1, entity_type=EntityType.RECOMMENDABLE, sentiment=Sentiment.UNKNOWN),
        CountFilter(lambda count: count >= 1, entity_type=EntityType.DESCRIPTIVE, sentiment=Sentiment.ANY)
    ], ranking_options=RankingOptions(num_positive=1, num_unknown=1, num_negative=1, default_cutoff=3,
                                      sentiment_utility={Sentiment.POSITIVE: 1, Sentiment.UNKNOWN: 0.5}),
       validator=Validator(metric=Metric.TAU, cutoff=3), include_unknown=True)

default = ExperimentOptions(name='default', split_seeds=[42], count_filters=[
        CountFilter(lambda count: count >= 1, entity_type=EntityType.RECOMMENDABLE, sentiment=Sentiment.POSITIVE),
        CountFilter(lambda count: count >= 1, entity_type=EntityType.DESCRIPTIVE, sentiment=Sentiment.ANY)
    ], ranking_options=RankingOptions(num_positive=1, num_unseen=100), include_unknown=False)

experiments = [default, separation]

if __name__ == '__main__':
    logger.info(f'Working directory: {os.getcwd()}')

    if len(sys.argv) < 3:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    for experiment in experiments:
        partition_interview.partition(experiment, input_directory=args.input[0], output_directory=args.output[0])
