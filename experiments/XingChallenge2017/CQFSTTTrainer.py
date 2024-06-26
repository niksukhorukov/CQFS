from core.CQFSTTSampler import CQFSTTSampler
from data.DataLoader import XingChallenge2017Loader
from experiments.train_CQFSTT import train_CQFSTT
from recsys.Recommender_import_list import ItemKNNCFRecommender, PureSVDItemRecommender, RP3betaRecommender


def main():
    data_loader = XingChallenge2017Loader()
    ICM_name = 'ICM_all'

    parameter_product = True
    percentages = [40, 60, 80, 95]
    alphas = [1]
    betas = [1, 1e-1, 1e-2, 1e-3, 1e-4]
    combination_strengths = [1, 10, 100, 1000, 10000]

    CF_recommender_classes = [ItemKNNCFRecommender, PureSVDItemRecommender, RP3betaRecommender]
    sampler = CQFSTTSampler(rmax=2, evals=1e4)

    cpu_count_div = 1
    cpu_count_sub = 0

    train_CQFSTT(
        data_loader=data_loader, ICM_name=ICM_name,
        percentages=percentages, alphas=alphas, betas=betas,
        combination_strengths=combination_strengths,
        CF_recommender_classes=CF_recommender_classes,
        cpu_count_div=cpu_count_div, cpu_count_sub=cpu_count_sub,
        sampler=sampler, parameter_product=parameter_product,
    )


if __name__ == '__main__':
    main()
