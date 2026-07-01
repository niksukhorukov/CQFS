import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from experiments.runtime_options import parse_runtime_options, print_runtime_options, runtime_kwargs


def main():
    args = parse_runtime_options("Run XingChallenge2017 TFIDF CBF baseline.")
    print_runtime_options(args)

    from data.DataLoader import XingChallenge2017Loader
    from experiments.baseline_TFIDF import baseline_TFIDF

    data_loader = XingChallenge2017Loader()
    ICM_name = 'ICM_all'
    baseline_TFIDF(data_loader, ICM_name, **runtime_kwargs(args))


if __name__ == "__main__":
    main()
