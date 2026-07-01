import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from experiments.runtime_options import parse_runtime_options, print_runtime_options, runtime_kwargs


def main():
    args = parse_runtime_options("Run CiteULike-a TFIDF CBF baseline.")
    print_runtime_options(args)

    from data.DataLoader import CiteULike_aLoader
    from experiments.baseline_TFIDF import baseline_TFIDF

    data_loader = CiteULike_aLoader()
    ICM_name = 'ICM_title_abstract'
    baseline_TFIDF(data_loader, ICM_name, **runtime_kwargs(args))


if __name__ == "__main__":
    main()
