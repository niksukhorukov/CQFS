import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(SCRIPT_DIR)

from experiments.runtime_options import parse_runtime_options, print_runtime_options, runtime_kwargs


def main():
    args = parse_runtime_options("Run XingChallenge2017 CBF baseline.")
    print_runtime_options(args)

    from data.DataLoader import XingChallenge2017Loader
    from experiments.baseline_CBF import baseline_CBF

    data_loader = XingChallenge2017Loader()
    ICM_name = 'ICM_all'
    baseline_CBF(data_loader, ICM_name, **runtime_kwargs(args))


if __name__ == "__main__":
    main()
