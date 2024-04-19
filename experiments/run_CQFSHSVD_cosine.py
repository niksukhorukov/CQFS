import numpy as np
from recsys.Base.Similarity.Cython.Compute_Similarity_Cython import Compute_Similarity_Cython

from core.CQFSHSVD_cosine import CQFSHSVD_cosine
from data.DataLoader import DataLoader
from recsys.Base.DataIO import DataIO
from recsys.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from utils.sparse import merge_sparse_matrices
from utils.statistics import warm_similarity_statistics
from scipy.sparse import diags
import numpy as np


def run_CQFSHSVD_cosine(
        *, data_loader: DataLoader, ICM_name, percentages, alphas, ranks, degs,
        CF_recommender_classes, sampler, save_FPMs=False,
        parameter_product=True, parameter_per_recommender=False,
        feature_weighting='tfidf'
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
    feature_to_index_mapper=data_loader.get_feature_to_index_mapper_from_name(ICM_name)
    
    
    ICM_train_reweighted = ICM_train.copy()
    if feature_weighting == 'tfidf_collab':
        popularity = np.array(URM_train.sum(axis=0)).squeeze()
        idf = np.log(ICM_train.shape[0] / (1.0 + np.array(ICM_train.sum(axis=0))).squeeze()) + 1.0
        ICM_train_reweighted = diags(popularity) @ ICM_train @ diags(idf)
        print('ICM tfidf_collab')
    
    if feature_weighting == 'tfidf_collab_new':
        popularity = np.array(URM_train.sum(axis=0)).squeeze()
        idf = np.log(ICM_train.shape[0] / (1.0 + np.array((diags(popularity) @ ICM_train).sum(axis=0))).squeeze()) + 1.0
        ICM_train_reweighted = diags(popularity) @ ICM_train @ diags(idf)
        print('ICM tfidf_collab_new')
    
    if feature_weighting == 'tfidf':
        idf = np.log(ICM_train.shape[0] / (1.0 + np.array(ICM_train.sum(axis=0))).squeeze()) + 1.0
        ICM_train_reweighted = ICM_train @ diags(idf)
        print('ICM tfidf')
    
    
    if ICM_name == 'ICM_metadata' and feature_weighting=='group':
        feats = {
            'ICM_metadata': ['genre_', 'production_company_', 'original_language_', 'release_date_', 'production_country_', 'spoken_lang_', 'collection_', 'ADULTS_', 'status_', 'status_', 'VIDEO_'],
            # 'ICM_all': ['region_', 'industry_', 'discipline_', 'country_', 'employment_', 'is_paid_', 'career_level_'],
            }

        features_to_columns = {}
        for key in feature_to_index_mapper:
            for feature in feats[ICM_name]:
                if key.startswith(feature):
                    if feature not in features_to_columns:
                        features_to_columns[feature] = []
                    features_to_columns[feature].append(data_loader.get_feature_to_index_mapper_from_name(ICM_name)[key])
                    
        ICM_train_reweighted = ICM_train.copy()
        for feature in features_to_columns:
            indices = features_to_columns[feature]
            feature_frequency = np.array(ICM_train[:, indices].sum(axis=1)).squeeze() ** 0.5
            inverse = np.divide(
                np.ones_like(feature_frequency),
                feature_frequency,
                out=np.ones_like(feature_frequency),
                where=feature_frequency != 0.0)
            ICM_train_reweighted[:, indices] = diags(feature_frequency ** -0.5) @ ICM_train_reweighted[:, indices]
        print('ICM group')
        

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
        CQFS_selector = CQFSHSVD_cosine(ICM_train_reweighted, URM_train, base_folder_path, sampler=sampler)

        ##################################################
        # Perform CQFS

        CQFS_selector.select_many_p(percentages, alphas, ranks, degs, save_FPMs=save_FPMs,
                                    parameter_product=parameter_product)
