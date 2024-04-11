import numpy as np
from recsys.Base.Similarity.Cython.Compute_Similarity_Cython import Compute_Similarity_Cython

from core.CQFSHSVD import CQFSHSVD
from data.DataLoader import DataLoader
from recsys.Base.DataIO import DataIO
from recsys.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from utils.sparse import merge_sparse_matrices
from utils.statistics import warm_similarity_statistics


def run_CQFSHSVD(
        *, data_loader: DataLoader, ICM_name, percentages, alphas, ranks, degs,
        CF_recommender_classes, sampler, save_FPMs=False,
        parameter_product=True, parameter_per_recommender=False,
):
    ##################################################
    # Data loading and splitting

    # Load data
    data_loader.load_data()
    dataset_name = data_loader.get_dataset_name()

    # Get the cold split
    URM_train, URM_validation, URM_test = data_loader.get_cold_split()
    print('rating shapes:', URM_train.shape, URM_validation.shape, URM_test.shape)

    # Create the last test URM by merging the train and validation matrices
    URM_train_validation = merge_sparse_matrices(URM_train, URM_validation).tocsr()

    ##################################################
    # ICM preparation

    ICM_train, original_ICM_train = data_loader.get_ICM_train_from_name(ICM_name, return_original=True)
    n_items, n_features = ICM_train.shape
    print(f"Training ICM has {n_items} items and {n_features} features.")

    ##################################################
    # Quantum Feature Selection

    # The CBF similarity used for the selection is a simple dot product
    topK = n_items
    # topK = 0
    # print("Computing similarity")
    # CBF_Similarity = Compute_Similarity_Cython(ICM_train.T, topK=topK, shrink=0, normalize=False, similarity='cosine')
    # S_CBF_original = CBF_Similarity.compute_similarity()
    # print("Computed similarity")

    if parameter_per_recommender:
        assert len(percentages) == len(alphas) == len(ranks) == len(degs) == len(CF_recommender_classes)
        iterable = zip(percentages, alphas, ranks, degs, CF_recommender_classes)
    else:
        iterable = (
            (percentages, alphas, ranks, degs, CF_recommender_class)
            for CF_recommender_class in CF_recommender_classes
        )

    for percentages, alphas, ranks, degs, CF_recommender_class in iterable:
        ##################################################
        # Setup collaborative filtering recommender

        # Get Collaborative Filtering best hyperparameters
        cf_recommender_name = CF_recommender_class.RECOMMENDER_NAME
        print(f"Loading collaborative model: {cf_recommender_name}")

        # cf_path = f"../../results/{dataset_name}/{cf_recommender_name}/"
        # cf_dataIO = DataIO(cf_path)

        # cf_similarity = "cosine_" if CF_recommender_class is ItemKNNCFRecommender else ""
        # cf_dict = cf_dataIO.load_data(f"{cf_recommender_name}_{cf_similarity}metadata.zip")
        # cf_best_hyperparameters = cf_dict['hyperparameters_best']

        # # Create Collaborative Filtering Recommender and fit with the best hyperparameters
        # CF_recommender = CF_recommender_class(URM_train_validation)
        # CF_recommender.fit(**cf_best_hyperparameters)

        # # Get CF and CBF Similarity Matrices
        # S_CF = CF_recommender.W_sparse.copy()
        # S_CBF = S_CBF_original.copy()

        # assert S_CF.shape == S_CBF.shape, "The two sparse matrices have different shapes!"
        # assert S_CF.shape == (n_items, n_items), "The similarity matrices do not have the right shape."

        # ##################################################
        # # Setup CQFS

        # # Get the warm items (CF) and the items with features (CBF)
        # CF_warm_items = np.ediff1d(URM_train_validation.tocsc().indptr) != 0
        # CBF_items_with_interactions = np.ediff1d(ICM_train.indptr) != 0

        # # Compute warm items statistics
        # statistics = warm_similarity_statistics(S_CF, S_CBF, CF_warm_items=CF_warm_items,
        #                                         CBF_items_with_interactions=CBF_items_with_interactions)

        base_folder_path = f"../../results/{dataset_name}/{ICM_name}/{cf_recommender_name}/"
        CQFS_selector = CQFSHSVD(ICM_train, URM_train, base_folder_path, sampler=sampler)

        ##################################################
        # Perform CQFS

        CQFS_selector.select_many_p(percentages, alphas, ranks, degs, save_FPMs=save_FPMs,
                                    parameter_product=parameter_product)
