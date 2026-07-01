import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from experiments.runtime_options import parse_runtime_options, print_runtime_options, runtime_kwargs


def main():
    args = parse_runtime_options("Run CiteULike-a collaborative filtering baselines.")
    print_runtime_options(args)

    from data.DataLoader import CiteULike_aLoader
    from experiments.train_CF import train_CF

    data_loader = CiteULike_aLoader()
    train_CF(data_loader, **runtime_kwargs(args))


if __name__ == '__main__':
    main()
