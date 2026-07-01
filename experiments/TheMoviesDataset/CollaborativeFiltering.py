import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(SCRIPT_DIR)

from experiments.runtime_options import parse_runtime_options, print_runtime_options, runtime_kwargs


def main():
    args = parse_runtime_options("Run TheMoviesDataset collaborative filtering baselines.")
    print_runtime_options(args)

    from data.DataLoader import TheMoviesDatasetLoader
    from experiments.train_CF import train_CF

    data_loader = TheMoviesDatasetLoader()
    train_CF(data_loader, **runtime_kwargs(args))


if __name__ == '__main__':
    main()
