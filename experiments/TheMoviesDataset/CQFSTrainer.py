import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(SCRIPT_DIR)

from experiments.runtime_options import parse_runtime_options, print_runtime_options, runtime_kwargs


def main():
    args = parse_runtime_options("Train TheMoviesDataset CQFS recommenders.")
    print_runtime_options(args)

    from data.DataLoader import TheMoviesDatasetLoader
    from experiments.train_CQFS import train_CQFS
    from recsys.Recommender_import_list import ItemKNNCFRecommender, PureSVDItemRecommender, \
        RP3betaRecommender

    from dwave.system import LeapHybridSampler
    # from neal import SimulatedAnnealingSampler
    # from core.CQFSSampler import CQFSSimulatedAnnealingSampler, CQFSQBSolvSampler

    data_loader = TheMoviesDatasetLoader()
    ICM_name = 'ICM_metadata'

    percentages = [40, 60, 80, 95]
    alphas = [1]
    betas = [1, 1e-1, 1e-2, 1e-3, 1e-4]
    combination_strengths = [1, 10, 100, 1000, 10000]

    solver_class = LeapHybridSampler
    # solver_class = SimulatedAnnealingSampler
    # solver_class = CQFSSimulatedAnnealingSampler
    # solver_class = CQFSQBSolvSampler

    CF_recommender_classes = [ItemKNNCFRecommender, PureSVDItemRecommender, RP3betaRecommender]

    cpu_count_div = 1
    cpu_count_sub = 0

    train_CQFS(data_loader, ICM_name, percentages, alphas, betas, combination_strengths, solver_class,
               CF_recommender_classes, cpu_count_div=cpu_count_div, cpu_count_sub=cpu_count_sub,
               **runtime_kwargs(args))


if __name__ == '__main__':
    main()
