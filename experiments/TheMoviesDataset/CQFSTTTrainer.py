from core.CQFSTTSampler import CQFSTTSampler
from data.DataLoader import TheMoviesDatasetLoader
from experiments.train_CQFSTT import train_CQFSTT
from recsys.Recommender_import_list import (
    ItemKNNCFRecommender, PureSVDItemRecommender, RP3betaRecommender,
)


def main():
    data_loader = TheMoviesDatasetLoader()
    ICM_name = 'ICM_metadata'

    parameter_product = True
    parameter_per_recommender = False
    percentages = [5, 10, 20, 30, 40, 60, 80, 95]
    alphas = [1]
    betas = [1, 1e-1, 1e-2, 1e-3, 1e-4]
    combination_strengths = [1, 10, 100, 1000, 10000]

    CF_recommender_classes = [
        ItemKNNCFRecommender,
        PureSVDItemRecommender,
        RP3betaRecommender,
    ]
    sampler = CQFSTTSampler(rmax=4, evals=1e6)

    cpu_count_div = 1
    cpu_count_sub = 0

    train_CQFSTT(
        data_loader=data_loader, ICM_name=ICM_name,
        percentages=percentages, alphas=alphas, betas=betas,
        combination_strengths=combination_strengths,
        CF_recommender_classes=CF_recommender_classes,
        cpu_count_div=cpu_count_div, cpu_count_sub=cpu_count_sub,
        sampler=sampler,
        parameter_product=parameter_product,
        parameter_per_recommender=parameter_per_recommender,
    )


if __name__ == '__main__':
    main()
