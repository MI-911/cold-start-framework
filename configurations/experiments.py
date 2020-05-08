from experiments.experiment import ExperimentOptions, CountFilter, RankingOptions
from shared.enums import Sentiment, Metric, EntityType, UnseenSampling
from shared.validator import Validator

separation = ExperimentOptions(name='separation', seed=123, count_filters=[
        CountFilter(lambda count: count >= 1, entity_type=EntityType.RECOMMENDABLE, sentiment=Sentiment.POSITIVE),
        CountFilter(lambda count: count >= 1, entity_type=EntityType.RECOMMENDABLE, sentiment=Sentiment.NEGATIVE),
        CountFilter(lambda count: count >= 1, entity_type=EntityType.RECOMMENDABLE, sentiment=Sentiment.UNKNOWN),
        CountFilter(lambda count: count >= 5, entity_type=EntityType.RECOMMENDABLE, sentiment=Sentiment.ANY),
        CountFilter(lambda count: count >= 5, entity_type=EntityType.DESCRIPTIVE, sentiment=Sentiment.ANY)
], ranking_options=RankingOptions(num_positive=1, num_unknown=1, num_negative=1, default_cutoff=3,
                                  sentiment_utility={Sentiment.POSITIVE: 1, Sentiment.UNKNOWN: 0.5}),
                               validator=Validator(metric=Metric.TAU, cutoff=3), include_unknown=True)

default = ExperimentOptions(name='default_uniform', seed=123, count_filters=[
        CountFilter(lambda count: count >= 5, entity_type=EntityType.DESCRIPTIVE, sentiment=Sentiment.ANY),
        CountFilter(lambda count: count >= 5, entity_type=EntityType.RECOMMENDABLE, sentiment=Sentiment.ANY),
        CountFilter(lambda count: count >= 1, entity_type=EntityType.RECOMMENDABLE, sentiment=Sentiment.POSITIVE)
    ], ranking_options=RankingOptions(unseen_sampling=UnseenSampling.UNIFORM, num_positive=1, num_unseen=100),
                            include_unknown=False, evaluation_samples=1)

movielens = ExperimentOptions(name='movielens', seed=123, count_filters=[
    CountFilter(lambda count: count >= 1, entity_type=EntityType.RECOMMENDABLE, sentiment=Sentiment.POSITIVE)
    ], ranking_options=RankingOptions(num_positive=1, num_unseen=100), include_unknown=False, evaluation_samples=1,
                              ratings_file='movielens.csv')

experiments = [default, separation]
experiment_names = [experiment.name for experiment in experiments]
