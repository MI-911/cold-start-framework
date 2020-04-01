import os
from enum import Enum
from typing import List, Callable

from experiments.data_loader import DataLoader


def sentiment_to_int(sentiment):
    return {
        Sentiment.NEGATIVE: -1,
        Sentiment.UNKNOWN: 0,
        Sentiment.POSITIVE: 1,
    }.get(sentiment, None)


class EntityType(Enum):
    RECOMMENDABLE = 1
    DESCRIPTIVE = 2
    ANY = 3


class Sentiment(Enum):
    NEGATIVE = 1
    UNKNOWN = 2
    POSITIVE = 3
    ANY = 4


class RankingOptions:
    def __init__(self, num_positive: int, num_negative: int = 0, num_unknown: int = 0, num_unseen: int = 0,
                 default_cutoff: int = 10):
        # Note discrepancy between 'unknown' and 'unseen', as 'unknown' is an explicit rating
        self.num_positive = num_positive
        self.num_negative = num_negative
        self.num_unknown = num_unknown
        self.num_unseen = num_unseen
        self.default_cutoff = default_cutoff

    def get_num_total(self):
        return self.num_positive + self.num_negative + self.num_unknown + self.num_unseen


class CountFilter:
    def __init__(self, filter_func: Callable[[int], bool], entity_type: EntityType,
                 sentiment: Sentiment = Sentiment.ANY):
        self.entity_type = entity_type
        self.sentiment = sentiment
        self.filter_func = filter_func


class ExperimentOptions:
    # TODO: Consider adding pre/post-operations on the ratings DF and warm/cold-start users
    def __init__(self, name: str, split_seeds: List[int], ranking_options: RankingOptions,
                 count_filters: List[CountFilter] = None, warm_start_ratio: float = 0.75,
                 include_unknown: bool = False, evaluation_samples: int = 10):
        self.name = name
        self.split_seeds = split_seeds
        self.count_filters = count_filters
        self.warm_start_ratio = warm_start_ratio
        self.include_unknown = include_unknown
        self.ranking_options = ranking_options
        self.evaluation_samples = evaluation_samples


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
