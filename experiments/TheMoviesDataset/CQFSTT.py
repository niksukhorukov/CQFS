from core.CQFSTTSampler import CQFSTTSampler
from data.DataLoader import TheMoviesDatasetLoader
from experiments.run_CQFSTT import run_CQFSTT
from recsys.Recommender_import_list import ItemKNNCFRecommender, PureSVDItemRecommender, RP3betaRecommender


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

    save_FPMs = False

    run_CQFSTT(
        data_loader=data_loader, ICM_name=ICM_name,
        percentages=percentages, alphas=alphas, betas=betas,
        combination_strengths=combination_strengths,
        CF_recommender_classes=CF_recommender_classes, sampler=sampler,
        save_FPMs=save_FPMs,
        parameter_product=parameter_product,
        parameter_per_recommender=parameter_per_recommender,
    )


if __name__ == '__main__':
    main()
