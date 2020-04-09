import os
from typing import List, Callable, Dict

from experiments.data_loader import DataLoader
from shared.enums import Sentiment, EntityType, Metric
from shared.validator import Validator


def sentiment_to_int(sentiment):
    return {
        Sentiment.NEGATIVE: -1,
        Sentiment.UNKNOWN: 0,
        Sentiment.POSITIVE: 1,
    }.get(sentiment, None)


class RankingOptions:
    def __init__(self, num_positive: int, num_negative: int = 0, num_unknown: int = 0, num_unseen: int = 0,
                 sentiment_utility: Dict[Sentiment, float] = None, default_cutoff: int = 10):
        # Note discrepancy between 'unknown' and 'unseen', as 'unknown' is an explicit rating
        self.sentiment_count = {Sentiment.POSITIVE: num_positive, Sentiment.NEGATIVE: num_negative,
                                Sentiment.UNKNOWN: num_unknown, Sentiment.UNSEEN: num_unseen}

        # Utility values for ranking, higher is better
        self.sentiment_utility = sentiment_utility if sentiment_utility else {Sentiment.POSITIVE: 1}

        # Default cutoffs are used in the recommenders' internal optimization
        self.default_cutoff = default_cutoff

        # Assert that the default cutoff does not exceed the total amount of samples
        assert self.default_cutoff <= self.get_num_total()

    def get_num_total(self):
        return sum(self.sentiment_count.values())


class CountFilter:
    def __init__(self, filter_func: Callable[[int], bool], entity_type: EntityType,
                 sentiment: Sentiment = Sentiment.ANY):
        self.entity_type = entity_type
        self.sentiment = sentiment
        self.filter_func = filter_func

    def __repr__(self):
        return f'{self.entity_type=}, {self.sentiment=}'


class ExperimentOptions:
    def __init__(self, name: str, split_seeds: List[int], ranking_options: RankingOptions,
                 count_filters: List[CountFilter] = None, warm_start_ratio: float = 0.75,
                 include_unknown: bool = False, evaluation_samples: int = 10,
                 validator: Validator = None):
        self.name = name
        self.split_seeds = split_seeds
        self.count_filters = count_filters
        self.warm_start_ratio = warm_start_ratio
        self.include_unknown = include_unknown
        self.ranking_options = ranking_options
        self.evaluation_samples = evaluation_samples
        self.validator = validator if validator else Validator(metric=Metric.NDCG, cutoff=10)


class Dataset:
    def __init__(self, path: str, experiments: List[str] = None):
        if not os.path.exists(path):
            raise IOError(f'Dataset path {path} does not exist')

        self.name = os.path.basename(path)
        self.experiment_paths = []

        for item in os.listdir(path):
            full_path = os.path.join(path, item)

            if not os.path.isdir(full_path) or experiments and item not in experiments:
                continue

            self.experiment_paths.append(full_path)

        if not self.experiment_paths:
            raise RuntimeError(f'Dataset path {path} contains no experiments')

    def __str__(self):
        return self.name

    def experiments(self):
        for path in self.experiment_paths:
            yield Experiment(self, path)


class Experiment:
    def __init__(self, parent, path):
        if not os.path.exists(path):
            raise IOError(f'Experiment path {path} does not exist')

        self.path = path
        self.dataset = parent
        self.name = os.path.basename(path)
        self.split_paths = []

        for directory in os.listdir(path):
            full_path = os.path.join(path, directory)
            if not os.path.isdir(full_path):
                continue

            self.split_paths.append(full_path)

        if not self.split_paths:
            raise RuntimeError(f'Experiment path {path} contains no splits')

        self.split_paths = sorted(self.split_paths)

    def __str__(self):
        return f'{self.dataset}/{self.name}'

    def splits(self):
        for path in self.split_paths:
            yield Split(self, path)


class Split:
    def __init__(self, parent, path):
        file_seen = {file: False for file in ['training.pkl', 'testing.pkl', 'meta.pkl']}

        if not os.path.exists(path):
            raise IOError(f'Split path {path} does not exist')

        for file in os.listdir(path):
            if file in file_seen:
                file_seen[file] = True

        if not all(file_seen.values()):
            raise IOError(f'Split path {path} is missing at least one required file')

        self.experiment = parent
        self.name = os.path.basename(path)
        self.data_loader = DataLoader(path)

    def __str__(self):
        return f'{self.experiment}/{self.name}'
