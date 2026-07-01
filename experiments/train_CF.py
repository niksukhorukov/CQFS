from data.DataLoader import DataLoader
from experiments.runtime_options import make_validation_evaluator
from recsys.GraphBased.RP3betaRecommender import RP3betaRecommender
from recsys.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from recsys.MatrixFactorization.PureSVDRecommender import PureSVDItemRecommender
from recsys.ParameterTuning.run_parameter_search import runParameterSearch_Collaborative


def train_CF(data_loader: DataLoader, n_cases=50, n_random_starts=15,
             use_fast_validation_evaluator=True, enable_similarity_cache=True,
             similarity_cache_memory_mb=2048):
    # Load data
    data_loader.load_data()
    dataset_name = data_loader.get_dataset_name()

    # Get the warm split
    URM_train, URM_validation, URM_test = data_loader.get_warm_split()

    # Instantiate the validation evaluator needed by the parameter search algorithm
    evaluator_validation = make_validation_evaluator(
        URM_validation,
        cutoff_list=[10],
        use_fast_validation_evaluator=use_fast_validation_evaluator,
    )

    recommender_classes = [ItemKNNCFRecommender, PureSVDItemRecommender, RP3betaRecommender]
    for Recommender in recommender_classes:
        # Name of the experiment and output results folder path
        recommendation_folder = f"{dataset_name}/{Recommender.RECOMMENDER_NAME}"
        output_folder_path = f"../../results/{recommendation_folder}/"

        runParameterSearch_Collaborative(Recommender, URM_train, evaluator_validation=evaluator_validation,
                                         output_folder_path=output_folder_path, n_cases=n_cases,
                                         n_random_starts=n_random_starts, resume_from_saved=True,
                                         enable_similarity_cache=enable_similarity_cache,
                                         similarity_cache_memory_mb=similarity_cache_memory_mb)
