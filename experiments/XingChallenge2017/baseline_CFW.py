import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(SCRIPT_DIR)

from experiments.runtime_options import parse_runtime_options, print_runtime_options, runtime_kwargs


def main():
    args = parse_runtime_options("Run XingChallenge2017 CFW baseline.", include_similarity_cache=False)
    print_runtime_options(args, include_similarity_cache=False)

    from data.DataLoader import XingChallenge2017Loader
    from experiments.baseline_CFW import baseline_CFW
    from recsys.GraphBased.RP3betaRecommender import RP3betaRecommender
    from recsys.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
    from recsys.MatrixFactorization.PureSVDRecommender import PureSVDItemRecommender

    data_loader = XingChallenge2017Loader()
    ICM_name = 'ICM_all'
    CF_recommenders = [ItemKNNCFRecommender, PureSVDItemRecommender, RP3betaRecommender]
    baseline_CFW(data_loader, ICM_name, CF_recommenders,
                 **runtime_kwargs(args, include_similarity_cache=False))


if __name__ == "__main__":
    main()
