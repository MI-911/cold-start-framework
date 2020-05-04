import argparse
import os
import sys

from loguru import logger

from experiments.experiment import ExperimentOptions, CountFilter, RankingOptions
from partitioners import partition_interview
from shared.enums import EntityType, Sentiment, Metric
from shared.utility import valid_dir
from shared.validator import Validator

parser = argparse.ArgumentParser()
parser.add_argument('--input', nargs=1, type=valid_dir, help='path to sources/input data')
parser.add_argument('--output', nargs=1, type=valid_dir, help='path to output data')

separation = ExperimentOptions(name='separation', seed=123, count_filters=[
        CountFilter(lambda count: count >= 1, entity_type=EntityType.RECOMMENDABLE, sentiment=Sentiment.POSITIVE),
        CountFilter(lambda count: count >= 1, entity_type=EntityType.RECOMMENDABLE, sentiment=Sentiment.NEGATIVE),
        CountFilter(lambda count: count >= 1, entity_type=EntityType.RECOMMENDABLE, sentiment=Sentiment.UNKNOWN),
        CountFilter(lambda count: count >= 1, entity_type=EntityType.DESCRIPTIVE, sentiment=Sentiment.ANY)
    ], ranking_options=RankingOptions(num_positive=1, num_unknown=1, num_negative=1, default_cutoff=3,
                                      sentiment_utility={Sentiment.POSITIVE: 1, Sentiment.UNKNOWN: 0.5}),
                               validator=Validator(metric=Metric.TAU, cutoff=3), include_unknown=True)

default = ExperimentOptions(name='default', seed=123, count_filters=[
        CountFilter(lambda count: count >= 5, entity_type=EntityType.DESCRIPTIVE, sentiment=Sentiment.ANY),
        CountFilter(lambda count: count >= 5, entity_type=EntityType.RECOMMENDABLE, sentiment=Sentiment.ANY),
        CountFilter(lambda count: count >= 1, entity_type=EntityType.RECOMMENDABLE, sentiment=Sentiment.POSITIVE)
    ], ranking_options=RankingOptions(num_positive=1, num_unseen=100), include_unknown=False, evaluation_samples=1,)


movielens = ExperimentOptions(name='movielens', seed=123, count_filters=[
        CountFilter(lambda count: count >= 1, entity_type=EntityType.RECOMMENDABLE, sentiment=Sentiment.POSITIVE)
    ], ranking_options=RankingOptions(num_positive=1, num_unseen=100), include_unknown=False, evaluation_samples=1,
                              ratings_file='movielens.csv')

experiments = [default]

if __name__ == '__main__':
    logger.info(f'Working directory: {os.getcwd()}')

    if len(sys.argv) < 3:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    for experiment in experiments:
        partition_interview.partition(experiment, input_directory=args.input[0], output_directory=args.output[0])
