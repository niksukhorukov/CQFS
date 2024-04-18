from core.CQFSHSVDSampler import CQFSHSVDSampler
from data.DataLoader import TheMoviesDatasetLoader
from experiments.train_CQFSHSVD import train_CQFSHSVD
from recsys.Recommender_import_list import (
    ItemKNNCFRecommender, PureSVDItemRecommender, RP3betaRecommender,
)


def main():
    data_loader = TheMoviesDatasetLoader()
    ICM_name = 'ICM_metadata'

    parameter_product = True
    parameter_per_recommender = False
    percentages = [5, 10, 20, 30, 40, 60, 80, 95]
    # alphas = [0.9, 0.7, 0.5, 0.3, 0.1]
    alphas = [0.3]
    ranks = [100, 200, 400]
    degs = [0.5, 1.0, 1.5, 2.0]

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
