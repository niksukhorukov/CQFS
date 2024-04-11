import time

import numpy as np

from data.DataLoader import DataLoader
from recsys.Base.DataIO import DataIO
from recsys.Base.Evaluation.Evaluator import EvaluatorHoldout
from recsys.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from recsys.ParameterTuning.run_parameter_search import runParameterSearch_Content
from utils.multithreading import parallelize_function
from utils.recsys import test_ICM_feature_selection
from utils.sparse import select_columns, merge_sparse_matrices


def train_random_KNN(n_features, percentage, IDF_argsort, dataset_name, ICM_name, evaluator_validation, evaluator_test,
                    URM_train, URM_train_last_test, ICM_train, ICM_train_last_test, n_cases, n_random_starts,
                    similarity_type_list, seed):
    selection_time = time.time()
    k_features = round(n_features * percentage / 100)
    selection = IDF_argsort[:k_features]
    bool_selection = np.zeros(n_features, dtype=bool)
    bool_selection[selection] = True
    selection_time = time.time() - selection_time

    new_ICM = select_columns(ICM_train, bool_selection)
    test_ICM_feature_selection(new_ICM, bool_selection)

    new_ICM_train_last_test = select_columns(ICM_train_last_test, bool_selection)
    test_ICM_feature_selection(new_ICM_train_last_test, bool_selection)

    base_folder_path = f"../../results/{dataset_name}/{ICM_name}/random_s{seed}/p{percentage:03d}/"

    selection_timings = {
        'selection_time': selection_time,
    }

    selection_statistics = {
        'n_features': n_features,
        'k_percentage': percentage,
        'k_selected': len(selection),
        'random_selection': selection,
        'random_bool_selection': bool_selection,
    }

    dataIO = DataIO(base_folder_path)
    dataIO.save_data("timings", selection_timings)
    dataIO.save_data("statistics", selection_statistics)

    recommender_folder_path = f"{base_folder_path}/{ItemKNNCBFRecommender.RECOMMENDER_NAME}/"
    runParameterSearch_Content(ItemKNNCBFRecommender, URM_train, new_ICM, ICM_name,
                               URM_train_last_test=URM_train_last_test, ICM_last_test=new_ICM_train_last_test,
                               n_cases=n_cases, n_random_starts=n_random_starts, resume_from_saved=True,
                               save_model='best', evaluator_validation=evaluator_validation,
                               evaluator_test=evaluator_test, output_folder_path=recommender_folder_path,
                               similarity_type_list=similarity_type_list)


def baseline_random(data_loader: DataLoader, ICM_name, percentages, seed, n_cases=50, n_random_starts=15, similarity_type_list=['cosine'],
                   parallelize=True):
    ##################################################
    # Data loading and splitting

    # Load data
    data_loader.load_data()
    dataset_name = data_loader.get_dataset_name()

    # Get the split
    URM_train, URM_validation, URM_test = data_loader.get_cold_split()

    # Create the last test URM by merging the train and validation matrices
    URM_train_last_test = merge_sparse_matrices(URM_train, URM_validation).tocsr()

    ##################################################
    # ICM preparation

    # Get the original ICM
    ICM_train, original_ICM_train = data_loader.get_ICM_train_from_name(ICM_name, return_original=True)
    n_items, n_features = ICM_train.shape
    print(f"Training ICM has {n_items} items and {n_features} features.")

    ##################################################
    # Evaluators instantiation

    # Obtain the array of train item indices, to be ignored during validation
    train_warm_item_mask = np.ediff1d(URM_train.tocsc().indptr) != 0
    train_warm_item_mask = np.arange(len(train_warm_item_mask))[train_warm_item_mask]

    # Obtain the array of train and validation item indices, to be ignored during testing
    train_validation_warm_item_mask = np.ediff1d(URM_train_last_test.tocsc().indptr) != 0
    train_validation_warm_item_mask = np.arange(len(train_validation_warm_item_mask))[train_validation_warm_item_mask]

    # Create the evaluator objects for validation and test
    # Train items are ignored during validation; train and validation items are ignored during testing
    evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[10], ignore_items=train_warm_item_mask)
    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[5, 10, 20, 50],
                                      ignore_items=train_validation_warm_item_mask)

    ##################################################
    # TF-IDF index

    IDF_time = time.time()
    ICM_coo = ICM_train.astype(np.float32)
    ICM_coo = ICM_coo.tocoo()
    N = float(ICM_coo.shape[0])

    # Compute IDF
    # IDF = np.log(N / (1 + np.bincount(ICM_coo.col)))
    # IDF_argsort = np.argsort(-IDF)
    # IDF_time = time.time() - IDF_time
    random = np.random.RandomState(seed=seed).permutation(ICM_train.shape[1])

    timings = {
        'IDF_time': IDF_time,
    }

    statistics = {
        'n_features': n_features,
        'random': random,
    }

    random_folder_path = f"../../results/{dataset_name}/{ICM_name}/random_s{seed}/"
    dataIO = DataIO(random_folder_path)
    dataIO.save_data(f"timings_s{seed}", timings)
    dataIO.save_data(f"statistics_s{seed}", statistics)

    # percentages = [20, 40, 60, 80, 95]

    if parallelize:
        args = [(n_features, percentage, random, dataset_name, ICM_name, evaluator_validation, evaluator_test,
                 URM_train, URM_train_last_test, ICM_train, original_ICM_train, n_cases, n_random_starts,
                 similarity_type_list, seed)
                for percentage in percentages]
        parallelize_function(train_random_KNN, args, count_div=1, count_sub=0)

    else:
        for percentage in percentages:
            train_random_KNN(n_features, percentage, random, dataset_name, ICM_name, evaluator_validation,
                            evaluator_test, URM_train, URM_train_last_test, ICM_train, original_ICM_train, n_cases,
                            n_random_starts, similarity_type_list, seed)
