from core.CQFSHSVDSampler import CQFSHSVDSampler
from data.DataLoader import CiteULike_aLoader
from experiments.train_CQFSHSVD import train_CQFSHSVD
from recsys.Recommender_import_list import (
    ItemKNNCFRecommender, PureSVDItemRecommender, RP3betaRecommender,
)

from core.mkl import mkl_set_num_threads


def main():
    num_threads = 12
    mkl_set_num_threads(num_threads)
    data_loader = CiteULike_aLoader()
    ICM_name = 'ICM_title_abstract'

    parameter_product = True
    parameter_per_recommender = False
    percentages = [5, 20, 30, 40, 60, 80, 95]
    alphas = [0.1, 0.4, 0.7, 0.9]
    ranks = [100, 200, 400]
    degs = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5]

    CF_recommender_classes = [
        PureSVDItemRecommender,
    ]
    sampler = CQFSHSVDSampler()

    cpu_count_div = 1
    cpu_count_sub = 0

    train_CQFSHSVD(
        data_loader=data_loader, ICM_name=ICM_name,
        percentages=percentages, alphas=alphas, ranks=ranks,
        degs=degs,
        CF_recommender_classes=CF_recommender_classes,
        cpu_count_div=cpu_count_div, cpu_count_sub=cpu_count_sub,
        sampler=sampler,
        parameter_product=parameter_product,
        parameter_per_recommender=parameter_per_recommender,
    )


if __name__ == '__main__':
    main()
