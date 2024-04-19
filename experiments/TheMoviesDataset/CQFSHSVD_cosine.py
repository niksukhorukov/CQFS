from core.CQFSHSVDSampler import CQFSHSVDSampler
from data.DataLoader import TheMoviesDatasetLoader
from experiments.run_CQFSHSVD_cosine import run_CQFSHSVD_cosine
from recsys.Recommender_import_list import (
    ItemKNNCFRecommender,PureSVDItemRecommender, RP3betaRecommender,
)

def main():
    data_loader = TheMoviesDatasetLoader()
    ICM_name = 'ICM_metadata'

    parameter_product = True
    parameter_per_recommender = False
    percentages = [5, 10, 20, 30, 40, 60, 80, 95]
    alphas = [0.8, 0.5, 0.2]
    ranks = [100, 200, 400]
    degs = [-1.0, -0.5, 0.0, 0.5, 1.0]
    feature_weighting = 'tfidf'

    CF_recommender_classes = [
        PureSVDItemRecommender,
    ]
    sampler = CQFSHSVDSampler()

    save_FPMs = False

    run_CQFSHSVD_cosine(
        data_loader=data_loader, ICM_name=ICM_name,
        percentages=percentages, alphas=alphas, ranks=ranks,
        degs=degs,
        CF_recommender_classes=CF_recommender_classes, sampler=sampler,
        save_FPMs=save_FPMs,
        parameter_product=parameter_product,
        parameter_per_recommender=parameter_per_recommender,
        feature_weighting=feature_weighting
    )


if __name__ == '__main__':
    main()
