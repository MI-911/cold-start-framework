import argparse
import subprocess

from configurations.models import models

parser = argparse.ArgumentParser()
parser.add_argument('--include', nargs='*', type=str, choices=models.keys(), help='models to include')
parser.add_argument('--experiments', nargs='*', type=str, default=['default'], help='experiments to run')

if __name__ == '__main__':
    args = parser.parse_args()
    processes = []
    for model in args.include:
        tmp = ['python3.8', 'interview.py', '--include', model, '--experiment', *args.experiments]
        processes.append(subprocess.Popen(['python3.8', 'interview.py', '--include', model, '--experiment',
                                           *args.experiments], stdout=None, stderr=subprocess.STDOUT))

    [p.wait() for p in processes]
