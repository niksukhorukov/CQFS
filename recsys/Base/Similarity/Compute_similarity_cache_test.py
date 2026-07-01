#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest

import numpy as np
import scipy.sparse as sps

from recsys.Base.Similarity.Compute_Similarity import Compute_Similarity
from recsys.Base.Similarity.SimilarityComputationCache import SimilarityComputationCache


class ComputeSimilarityCacheTest(unittest.TestCase):

    def setUp(self):
        dense_matrix = np.array([
            [1.0, 0.0, 2.0, 0.0, 3.0],
            [0.0, 4.0, 0.0, 5.0, 1.0],
            [2.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 1.0, 2.0],
            [1.0, 1.0, 0.0, 0.0, 4.0],
            [0.0, 2.0, 5.0, 1.0, 0.0],
        ], dtype=np.float32)
        self.data_matrix = sps.csr_matrix(dense_matrix)

    def assert_sparse_allclose(self, expected, actual):
        self.assertEqual(expected.shape, actual.shape)
        self.assertTrue(
            np.allclose(expected.toarray(), actual.toarray(), rtol=1e-5, atol=1e-6),
            "cached similarity does not match reference",
        )

    def compute_reference(self, **kwargs):
        similarity = Compute_Similarity(self.data_matrix, use_implementation="python", **kwargs)
        return similarity.compute_similarity()

    def compute_cached(self, cache, **kwargs):
        similarity = Compute_Similarity(
            self.data_matrix,
            use_implementation="python",
            _similarity_cache=cache,
            **kwargs
        )
        return similarity.compute_similarity()

    def compute_uncached_cython(self, **kwargs):
        try:
            from recsys.Base.Similarity.Cython.Compute_Similarity_Cython import Compute_Similarity_Cython
        except ImportError:
            self.skipTest("Cython similarity module is not available")

        similarity = Compute_Similarity_Cython(self.data_matrix, **kwargs)
        return similarity.compute_similarity()

    def compute_cached_cython(self, **kwargs):
        try:
            from recsys.Base.Similarity.Cython.Compute_Similarity_Cython import Compute_Similarity_Cached_Cython
        except ImportError:
            self.skipTest("Cython similarity module is not available")

        cache = SimilarityComputationCache(verbose=False)
        cached_data = cache.get_similarity_data(self.data_matrix, similarity=kwargs.get("similarity", "cosine"))
        similarity = Compute_Similarity_Cached_Cython(cached_data, **kwargs)
        return similarity.compute_similarity()

    def test_cached_cosine_reuses_raw_similarity_across_topk_and_shrink(self):
        cache = SimilarityComputationCache(verbose=False)

        expected = self.compute_reference(similarity="cosine", topK=2, shrink=0, normalize=True)
        actual = self.compute_cached(cache, similarity="cosine", topK=2, shrink=0, normalize=True)
        self.assert_sparse_allclose(expected, actual)
        self.assertEqual(cache.misses, 1)
        self.assertEqual(cache.hits, 0)

        expected = self.compute_reference(similarity="cosine", topK=4, shrink=7, normalize=False)
        actual = self.compute_cached(cache, similarity="cosine", topK=4, shrink=7, normalize=False)
        self.assert_sparse_allclose(expected, actual)
        self.assertEqual(cache.misses, 1)
        self.assertEqual(cache.hits, 1)

    def test_cached_similarity_modes_match_reference(self):
        test_cases = [
            {"similarity": "cosine", "topK": 3, "shrink": 0, "normalize": False},
            {"similarity": "asymmetric", "topK": 5, "shrink": 3, "normalize": True, "asymmetric_alpha": 0.3},
            {"similarity": "jaccard", "topK": 5, "shrink": 2, "normalize": True},
            {"similarity": "dice", "topK": 5, "shrink": 4, "normalize": True},
            {"similarity": "tversky", "topK": 5, "shrink": 1, "normalize": True, "tversky_alpha": 0.7, "tversky_beta": 1.3},
        ]

        for kwargs in test_cases:
            cache = SimilarityComputationCache(verbose=False)
            expected = self.compute_reference(**kwargs)
            actual = self.compute_cached(cache, **kwargs)
            self.assert_sparse_allclose(expected, actual)

    def test_cached_cython_similarity_modes_match_uncached_cython(self):
        test_cases = [
            {"similarity": "cosine", "topK": 2, "shrink": 0, "normalize": True},
            {"similarity": "cosine", "topK": 4, "shrink": 7, "normalize": False},
            {"similarity": "asymmetric", "topK": 4, "shrink": 3, "normalize": True, "asymmetric_alpha": 0.3},
            {"similarity": "jaccard", "topK": 5, "shrink": 2, "normalize": True},
            {"similarity": "dice", "topK": 5, "shrink": 4, "normalize": True},
            {"similarity": "tversky", "topK": 5, "shrink": 1, "normalize": True, "tversky_alpha": 0.7, "tversky_beta": 1.3},
        ]

        for kwargs in test_cases:
            expected = self.compute_uncached_cython(**kwargs)
            actual = self.compute_cached_cython(**kwargs)
            self.assert_sparse_allclose(expected, actual)

    def test_cached_row_weights_fall_back_to_reference(self):
        row_weights = np.array([1.0, 0.5, 1.5, 0.25, 2.0, 0.75], dtype=np.float64)
        cache = SimilarityComputationCache(verbose=False)

        expected = self.compute_reference(
            similarity="cosine", topK=5, shrink=2, normalize=True, row_weights=row_weights
        )
        actual = self.compute_cached(
            cache, similarity="cosine", topK=5, shrink=2, normalize=True, row_weights=row_weights
        )

        self.assert_sparse_allclose(expected, actual)
        self.assertEqual(cache.misses, 0)
        self.assertEqual(cache.hits, 0)
        self.assertEqual(cache.disabled, 0)

    def test_topk_zero_does_not_build_cache(self):
        try:
            from recsys.Base.Similarity.Cython.Compute_Similarity_Cython import Compute_Similarity_Cython
        except ImportError:
            self.skipTest("Cython similarity module is not available")

        cache = SimilarityComputationCache(verbose=False)

        similarity = Compute_Similarity(
            self.data_matrix,
            use_implementation="cython",
            _similarity_cache=cache,
            similarity="cosine",
            topK=0,
            shrink=0,
            normalize=True,
        )
        result = similarity.compute_similarity()

        self.assertEqual(result.shape, (self.data_matrix.shape[1], self.data_matrix.shape[1]))
        self.assertEqual(cache.misses, 0)
        self.assertEqual(cache.hits, 0)
        self.assertEqual(cache.disabled, 0)

    def test_cache_budget_falls_back_to_reference(self):
        cache = SimilarityComputationCache(max_memory_bytes=1, verbose=False)

        expected = self.compute_reference(similarity="cosine", topK=3, shrink=0, normalize=True)
        actual = self.compute_cached(cache, similarity="cosine", topK=3, shrink=0, normalize=True)

        self.assert_sparse_allclose(expected, actual)
        self.assertEqual(cache.disabled, 1)
        self.assertEqual(cache.misses, 1)
        self.assertEqual(cache.hits, 0)


if __name__ == "__main__":
    unittest.main()
