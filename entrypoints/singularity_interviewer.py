import argparse
import subprocess

from configurations.models import models

parser = argparse.ArgumentParser()
parser.add_argument('--include', nargs='*', type=str, choices=models.keys(), help='models to include')
parser.add_argument('--experiments', nargs='*', type=str, default=['default'], help='experiments to run')

if __name__ == '__main__':
    args, unknown_args = parser.parse_known_args()
    processes = []
    for model in args.include:
        tmp = ['python3.8', 'interview.py', '--include', model, '--experiment', *args.experiments, *unknown_args]
        processes.append(subprocess.Popen(tmp, stdout=None, stderr=subprocess.STDOUT))

    print('Waiting for processes')
    [p.wait() for p in processes]
