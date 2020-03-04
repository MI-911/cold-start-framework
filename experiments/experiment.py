import os
from typing import List

from experiments.data_loader import DataLoader


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
