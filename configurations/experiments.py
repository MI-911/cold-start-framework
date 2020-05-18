from copy import deepcopy

from experiments.experiment import ExperimentOptions, CountFilter, RankingOptions
from shared.enums import Sentiment, Metric, EntityType, UnseenSampling, SeenSampling
from shared.validator import Validator

separation = ExperimentOptions(name='separation', seed=42, count_filters=[
        CountFilter(lambda count: count >= 1, entity_type=EntityType.RECOMMENDABLE, sentiment=Sentiment.POSITIVE),
        CountFilter(lambda count: count >= 1, entity_type=EntityType.RECOMMENDABLE, sentiment=Sentiment.NEGATIVE),
        CountFilter(lambda count: count >= 5, entity_type=EntityType.RECOMMENDABLE, sentiment=Sentiment.ANY),
        CountFilter(lambda count: count >= 5, entity_type=EntityType.DESCRIPTIVE, sentiment=Sentiment.ANY)
], ranking_options=RankingOptions(num_positive=1, num_negative=1, num_unknown=1, default_cutoff=3,
                                  sentiment_utility={Sentiment.POSITIVE: 1, Sentiment.UNKNOWN: 0.5}),
                               validator=Validator(metric=Metric.TAU, cutoff=3), include_unknown=True)

default = ExperimentOptions(name='default_uniform', seed=42, count_filters=[
        CountFilter(lambda count: count >= 5, entity_type=EntityType.DESCRIPTIVE, sentiment=Sentiment.ANY),
        CountFilter(lambda count: count >= 5, entity_type=EntityType.RECOMMENDABLE, sentiment=Sentiment.ANY),
        CountFilter(lambda count: count >= 1, entity_type=EntityType.RECOMMENDABLE, sentiment=Sentiment.POSITIVE)
    ], ranking_options=RankingOptions(unseen_sampling=UnseenSampling.UNIFORM, num_positive=1, num_unseen=100),
                            include_unknown=False, evaluation_samples=1)

movielens = ExperimentOptions(name='movielens', seed=42, count_filters=[
    CountFilter(lambda count: count >= 1, entity_type=EntityType.RECOMMENDABLE, sentiment=Sentiment.POSITIVE)
    ], ranking_options=RankingOptions(num_positive=1, num_unseen=100), include_unknown=False, evaluation_samples=1,
                              ratings_file='movielens.csv')

default_equal_probability = deepcopy(default)
default_equal_probability.name = 'default_equal_popularity'
default_equal_probability.ranking_options.unseen_sampling = UnseenSampling.EQUAL_POPULARITY

default_long_tail = deepcopy(default)
default_long_tail.name = 'default_long_tail'
default_long_tail.ranking_options.seen_sampling = SeenSampling.LONG_TAIL

experiments = [default, separation, default_equal_probability, default_long_tail]
experiment_names = [experiment.name for experiment in experiments]
