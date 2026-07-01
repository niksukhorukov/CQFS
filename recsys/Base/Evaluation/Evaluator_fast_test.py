#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest

import numpy as np

for alias_name, alias_value in {"bool": bool, "float": float, "int": int}.items():
    if alias_name not in np.__dict__:
        setattr(np, alias_name, alias_value)

import scipy.sparse as sps

from recsys.Base.BaseRecommender import BaseRecommender
from recsys.Base.Evaluation.Evaluator import EvaluatorHoldout, EvaluatorHoldoutFast


class FixedScoreRecommender(BaseRecommender):
    RECOMMENDER_NAME = "FixedScoreRecommender"

    def __init__(self, URM_train, score_matrix):
        super(FixedScoreRecommender, self).__init__(URM_train, verbose=False)
        self.score_matrix = np.asarray(score_matrix, dtype=np.float32)

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        scores = self.score_matrix[user_id_array].copy()

        if items_to_compute is not None:
            filtered_scores = -np.ones_like(scores) * np.inf
            filtered_scores[:, items_to_compute] = scores[:, items_to_compute]
            scores = filtered_scores

        return scores


class EvaluatorHoldoutFastTest(unittest.TestCase):

    def setUp(self):
        train_dense = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1],
            [1, 0, 1, 0, 0, 0, 0, 0],
        ], dtype=np.float32)

        test_dense = np.array([
            [0, 4, 2, 0, 0, 0, 0, 0],
            [3, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 5, 0, 2, 0, 0],
            [1, 0, 0, 0, 3, 0, 0, 0],
            [0, 2, 0, 0, 0, 0, 4, 1],
        ], dtype=np.float32)

        self.URM_train = sps.csr_matrix(train_dense)
        self.URM_test = sps.csr_matrix(test_dense)
        self.score_matrix = np.array([
            [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2],
            [0.4, 0.9, 0.2, 0.8, 0.7, 0.1, 0.6, 0.3],
            [0.1, 0.3, 0.9, 0.8, 0.2, 0.7, 0.4, 0.6],
            [0.6, 0.5, 0.4, 0.9, 0.8, 0.7, 0.3, 0.1],
            [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2],
        ], dtype=np.float32)

    def _assert_fast_matches_full(self, full_results, fast_results):
        for cutoff in fast_results:
            self.assertIn(cutoff, full_results)

            for metric_name in fast_results[cutoff]:
                self.assertIn(metric_name, full_results[cutoff])
                self.assertTrue(
                    np.allclose(full_results[cutoff][metric_name], fast_results[cutoff][metric_name], atol=1e-10),
                    "{}@{} differs: full {}, fast {}".format(
                        metric_name,
                        cutoff,
                        full_results[cutoff][metric_name],
                        fast_results[cutoff][metric_name],
                    )
                )

    def test_recommend_batch_array_matches_recommend_lists(self):
        recommender = FixedScoreRecommender(self.URM_train, self.score_matrix)

        ranking_list = recommender.recommend(np.arange(self.URM_train.shape[0]), cutoff=5, remove_seen_flag=True)
        ranking_array = recommender.recommend_batch_array(np.arange(self.URM_train.shape[0]), cutoff=5,
                                                          remove_seen_flag=True)

        for user_index, user_ranking_array in enumerate(ranking_array):
            self.assertEqual(ranking_list[user_index], user_ranking_array[user_ranking_array >= 0].tolist())

    def test_fast_matches_full_multiple_cutoffs(self):
        recommender = FixedScoreRecommender(self.URM_train, self.score_matrix)

        full_evaluator = EvaluatorHoldout(self.URM_test, cutoff_list=[1, 3, 5], verbose=False)
        fast_evaluator = EvaluatorHoldoutFast(self.URM_test, cutoff_list=[1, 3, 5], verbose=False)

        full_results, _ = full_evaluator.evaluateRecommender(recommender)
        fast_results, _ = fast_evaluator.evaluateRecommender(recommender)

        self._assert_fast_matches_full(full_results, fast_results)

    def test_fast_matches_full_with_ignore_filters_and_min_ratings(self):
        recommender = FixedScoreRecommender(self.URM_train, self.score_matrix)

        full_evaluator = EvaluatorHoldout(
            self.URM_test,
            cutoff_list=[2, 4],
            min_ratings_per_user=2,
            ignore_items=[5],
            ignore_users=[1],
            verbose=False,
        )
        fast_evaluator = EvaluatorHoldoutFast(
            self.URM_test,
            cutoff_list=[2, 4],
            min_ratings_per_user=2,
            ignore_items=[5],
            ignore_users=[1],
            verbose=False,
        )

        full_results, _ = full_evaluator.evaluateRecommender(recommender)
        fast_results, _ = fast_evaluator.evaluateRecommender(recommender)

        self._assert_fast_matches_full(full_results, fast_results)

    def test_metric_subset_matches_full(self):
        recommender = FixedScoreRecommender(self.URM_train, self.score_matrix)
        metrics_to_compute = ["MAP", "PRECISION", "RECALL", "NDCG"]

        full_evaluator = EvaluatorHoldout(self.URM_test, cutoff_list=[5], verbose=False)
        fast_evaluator = EvaluatorHoldoutFast(self.URM_test, cutoff_list=[5], verbose=False,
                                              metrics_to_compute=metrics_to_compute)

        full_results, _ = full_evaluator.evaluateRecommender(recommender)
        fast_results, _ = fast_evaluator.evaluateRecommender(recommender)

        self.assertEqual(set(metrics_to_compute), set(fast_results[5].keys()))
        self._assert_fast_matches_full(full_results, fast_results)


if __name__ == "__main__":
    unittest.main()
