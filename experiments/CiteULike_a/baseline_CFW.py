import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from experiments.runtime_options import parse_runtime_options, print_runtime_options, runtime_kwargs


def main():
    args = parse_runtime_options("Run CiteULike-a CFW baseline.", include_similarity_cache=False)
    print_runtime_options(args, include_similarity_cache=False)

    from data.DataLoader import CiteULike_aLoader
    from experiments.baseline_CFW import baseline_CFW
    from recsys.GraphBased.RP3betaRecommender import RP3betaRecommender
    from recsys.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
    from recsys.MatrixFactorization.PureSVDRecommender import PureSVDItemRecommender

    data_loader = CiteULike_aLoader()
    ICM_name = 'ICM_title_abstract'
    CF_recommenders = [ItemKNNCFRecommender, PureSVDItemRecommender, RP3betaRecommender]
    baseline_CFW(data_loader, ICM_name, CF_recommenders,
                 **runtime_kwargs(args, include_similarity_cache=False))


if __name__ == "__main__":
    main()
