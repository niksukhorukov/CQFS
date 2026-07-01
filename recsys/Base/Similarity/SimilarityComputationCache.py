#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import hashlib
import time

import numpy as np
import scipy.sparse as sps

from recsys.Base.Recommender_utils import check_matrix


class SimilarityComputationCache(object):
    """In-memory cache for raw sparse similarity co-occurrences."""

    def __init__(self, max_memory_bytes=2 * 1024 ** 3, verbose=True):
        super(SimilarityComputationCache, self).__init__()

        self.max_memory_bytes = max_memory_bytes
        self.verbose = verbose
        self._cache = {}
        self.current_memory_bytes = 0
        self.hits = 0
        self.misses = 0
        self.disabled = 0

    def _print(self, message):
        if self.verbose:
            print("SimilarityComputationCache: {}".format(message))

    def get_similarity_data(self, dataMatrix, similarity="cosine", row_weights=None):
        if not sps.issparse(dataMatrix):
            self.disabled += 1
            return None

        if similarity in ["adjusted", "pearson"]:
            self.disabled += 1
            return None

        if row_weights is not None:
            self.disabled += 1
            return None

        key = self._get_cache_key(dataMatrix, similarity, row_weights)

        if key in self._cache:
            self.hits += 1
            return self._cache[key]

        self.misses += 1

        prepared_matrix = _prepare_similarity_matrix(dataMatrix, similarity)
        estimated_memory = _estimate_raw_similarity_memory(prepared_matrix)

        if self.current_memory_bytes + estimated_memory > self.max_memory_bytes:
            self.disabled += 1
            self._print(
                "disabled for this matrix, estimated raw cache {:.2f} MB exceeds remaining budget {:.2f} MB".format(
                    estimated_memory / 1024 ** 2,
                    (self.max_memory_bytes - self.current_memory_bytes) / 1024 ** 2,
                )
            )
            return None

        start_time = time.time()
        cached_data = _build_cached_similarity_data(prepared_matrix, similarity, row_weights)

        if self.current_memory_bytes + cached_data.memory_bytes > self.max_memory_bytes:
            self.disabled += 1
            self._print(
                "disabled for this matrix, actual raw cache {:.2f} MB exceeds remaining budget {:.2f} MB".format(
                    cached_data.memory_bytes / 1024 ** 2,
                    (self.max_memory_bytes - self.current_memory_bytes) / 1024 ** 2,
                )
            )
            return None

        self._cache[key] = cached_data
        self.current_memory_bytes += cached_data.memory_bytes

        self._print(
            "built raw cache with {} non-zero values ({:.2f} MB) in {:.2f} seconds".format(
                cached_data.raw_similarity.nnz,
                cached_data.memory_bytes / 1024 ** 2,
                time.time() - start_time,
            )
        )

        return cached_data

    def _get_cache_key(self, dataMatrix, similarity, row_weights):
        dataMatrix = check_matrix(dataMatrix, "csr")
        dataMatrix.sort_indices()

        digest = hashlib.blake2b(digest_size=20)
        digest.update(str(dataMatrix.shape).encode("ascii"))
        digest.update(str(dataMatrix.dtype).encode("ascii"))
        digest.update(dataMatrix.indptr.tobytes())
        digest.update(dataMatrix.indices.tobytes())
        digest.update(dataMatrix.data.tobytes())
        digest.update(str(similarity).encode("ascii"))

        if row_weights is not None:
            row_weights = np.asarray(row_weights, dtype=np.float64)
            digest.update(str(row_weights.shape).encode("ascii"))
            digest.update(row_weights.tobytes())

        return digest.hexdigest()


class CachedSimilarityData(object):
    def __init__(self, raw_similarity, sum_of_squared, n_columns):
        super(CachedSimilarityData, self).__init__()

        self.raw_similarity = check_matrix(raw_similarity, "csc", dtype=np.float64)
        self.raw_similarity.sort_indices()
        self.sum_of_squared = np.asarray(sum_of_squared, dtype=np.float64)
        self.n_columns = n_columns
        self.memory_bytes = self.raw_similarity.data.nbytes + \
                            self.raw_similarity.indices.nbytes + \
                            self.raw_similarity.indptr.nbytes + \
                            self.sum_of_squared.nbytes


class Compute_Similarity_Cached(object):
    def __init__(self, cached_data, topK=100, shrink=0, normalize=True,
                 asymmetric_alpha=0.5, tversky_alpha=1.0, tversky_beta=1.0,
                 similarity="cosine"):
        super(Compute_Similarity_Cached, self).__init__()

        try:
            from recsys.Base.Similarity.Cython.Compute_Similarity_Cython import Compute_Similarity_Cached_Cython

            self.compute_similarity_object = Compute_Similarity_Cached_Cython(
                cached_data,
                topK=topK,
                shrink=shrink,
                normalize=normalize,
                asymmetric_alpha=asymmetric_alpha,
                tversky_alpha=tversky_alpha,
                tversky_beta=tversky_beta,
                similarity=similarity,
            )

        except ImportError:
            self.compute_similarity_object = Compute_Similarity_Cached_Python(
                cached_data,
                topK=topK,
                shrink=shrink,
                normalize=normalize,
                asymmetric_alpha=asymmetric_alpha,
                tversky_alpha=tversky_alpha,
                tversky_beta=tversky_beta,
                similarity=similarity,
            )

    def compute_similarity(self, **args):
        return self.compute_similarity_object.compute_similarity(**args)


class Compute_Similarity_Cached_Python(object):
    def __init__(self, cached_data, topK=100, shrink=0, normalize=True,
                 asymmetric_alpha=0.5, tversky_alpha=1.0, tversky_beta=1.0,
                 similarity="cosine"):
        super(Compute_Similarity_Cached_Python, self).__init__()

        self.cached_data = cached_data
        self.TopK = min(topK, cached_data.n_columns)
        self.shrink = shrink
        self.normalize = normalize
        self.asymmetric_alpha = asymmetric_alpha
        self.tversky_alpha = tversky_alpha
        self.tversky_beta = tversky_beta

        self.adjusted_cosine = similarity == "adjusted"
        self.asymmetric_cosine = similarity == "asymmetric"
        self.pearson_correlation = similarity == "pearson"
        self.tanimoto_coefficient = similarity in ["jaccard", "tanimoto"]
        self.dice_coefficient = similarity == "dice"
        self.tversky_coefficient = similarity == "tversky"

        if self.tanimoto_coefficient or self.dice_coefficient or self.tversky_coefficient:
            self.normalize = False

        self.sumOfSquared = cached_data.sum_of_squared

        if self.asymmetric_cosine:
            self.sumOfSquared_to_1_minus_alpha = np.power(self.sumOfSquared, 2 * (1 - self.asymmetric_alpha))
            self.sumOfSquared_to_alpha = np.power(self.sumOfSquared, 2 * self.asymmetric_alpha)

    def compute_similarity(self, start_col=None, end_col=None):
        if self.TopK == 0:
            return None

        raw_similarity = self.cached_data.raw_similarity
        n_columns = self.cached_data.n_columns

        start_col_local = 0
        end_col_local = n_columns

        if start_col is not None and start_col > 0 and start_col < n_columns:
            start_col_local = start_col

        if end_col is not None and end_col > start_col_local and end_col < n_columns:
            end_col_local = end_col

        values = []
        rows = []
        cols = []

        for column_index in range(start_col_local, end_col_local):
            start_position = raw_similarity.indptr[column_index]
            end_position = raw_similarity.indptr[column_index + 1]

            column_rows = raw_similarity.indices[start_position:end_position]
            column_weights = raw_similarity.data[start_position:end_position]

            if len(column_weights) == 0:
                continue

            this_column_weights = self._apply_denominator(column_index, column_rows, column_weights)

            non_zero_mask = this_column_weights != 0.0
            if not np.any(non_zero_mask):
                continue

            this_column_weights = this_column_weights[non_zero_mask]
            column_rows = column_rows[non_zero_mask]

            local_topK = min(self.TopK, len(this_column_weights))

            if local_topK < len(this_column_weights):
                relevant_items_partition = (-this_column_weights).argpartition(local_topK - 1)[0:local_topK]
                relevant_items_partition_sorting = np.argsort(-this_column_weights[relevant_items_partition])
                top_k_idx = relevant_items_partition[relevant_items_partition_sorting]
            else:
                top_k_idx = np.argsort(-this_column_weights)

            selected_rows = column_rows[top_k_idx]
            selected_values = this_column_weights[top_k_idx]

            values.extend(selected_values)
            rows.extend(selected_rows)
            cols.extend(np.ones(len(selected_values), dtype=np.int32) * column_index)

        return sps.csr_matrix(
            (values, (rows, cols)),
            shape=(n_columns, n_columns),
            dtype=np.float32,
        )

    def _apply_denominator(self, column_index, column_rows, column_weights):
        this_column_weights = column_weights.astype(np.float64, copy=True)

        if self.normalize:
            if self.asymmetric_cosine:
                denominator = self.sumOfSquared_to_alpha[column_index] * \
                              self.sumOfSquared_to_1_minus_alpha[column_rows] + self.shrink + 1e-6
            else:
                denominator = self.sumOfSquared[column_index] * self.sumOfSquared[column_rows] + self.shrink + 1e-6

            this_column_weights = np.multiply(this_column_weights, 1.0 / denominator)

        elif self.tanimoto_coefficient:
            denominator = self.sumOfSquared[column_index] + self.sumOfSquared[column_rows] - \
                          this_column_weights + self.shrink + 1e-6
            this_column_weights = np.multiply(this_column_weights, 1.0 / denominator)

        elif self.dice_coefficient:
            denominator = self.sumOfSquared[column_index] + self.sumOfSquared[column_rows] + self.shrink + 1e-6
            this_column_weights = np.multiply(this_column_weights, 1.0 / denominator)

        elif self.tversky_coefficient:
            denominator = this_column_weights + \
                          (self.sumOfSquared[column_index] - this_column_weights) * self.tversky_alpha + \
                          (self.sumOfSquared[column_rows] - this_column_weights) * self.tversky_beta + \
                          self.shrink + 1e-6
            this_column_weights = np.multiply(this_column_weights, 1.0 / denominator)

        elif self.shrink != 0:
            this_column_weights = this_column_weights / self.shrink

        return this_column_weights


def _prepare_similarity_matrix(dataMatrix, similarity):
    dataMatrix = check_matrix(dataMatrix.copy(), "csr")

    if similarity == "adjusted":
        dataMatrix = _apply_adjusted_cosine(dataMatrix)
    elif similarity == "pearson":
        dataMatrix = _apply_pearson_correlation(dataMatrix)
    elif similarity in ["jaccard", "tanimoto", "dice", "tversky"]:
        dataMatrix.data = np.ones_like(dataMatrix.data)

    return check_matrix(dataMatrix, "csr")


def _build_cached_similarity_data(dataMatrix, similarity, row_weights):
    sum_of_squared = np.array(dataMatrix.power(2).sum(axis=0), dtype=np.float64).ravel()

    if similarity not in ["jaccard", "tanimoto", "dice", "tversky"]:
        sum_of_squared = np.sqrt(sum_of_squared)

    if row_weights is not None:
        row_weights = np.asarray(row_weights, dtype=np.float64)

        if dataMatrix.shape[0] != len(row_weights):
            raise ValueError("Cosine_Similarity: provided row_weights and dataMatrix have different number of rows."
                             "Row_weights has {} rows, dataMatrix has {}.".format(len(row_weights), dataMatrix.shape[0]))

        weighted_dataMatrix = dataMatrix.multiply(row_weights[:, None])
        raw_similarity = weighted_dataMatrix.T.dot(dataMatrix)
    else:
        raw_similarity = dataMatrix.T.dot(dataMatrix)

    raw_similarity = check_matrix(raw_similarity, "csc", dtype=np.float64)
    raw_similarity.setdiag(0.0)
    raw_similarity.eliminate_zeros()

    return CachedSimilarityData(raw_similarity, sum_of_squared, dataMatrix.shape[1])


def _estimate_raw_similarity_memory(dataMatrix):
    dataMatrix = check_matrix(dataMatrix, "csr")
    row_nnz = np.ediff1d(dataMatrix.indptr).astype(np.float64)
    upper_bound_nnz = np.sum(row_nnz * row_nnz)

    return upper_bound_nnz * 32.0


def _apply_adjusted_cosine(dataMatrix):
    dataMatrix = check_matrix(dataMatrix, "csr")

    interactions_per_row = np.ediff1d(dataMatrix.indptr)
    nonzero_rows = interactions_per_row > 0
    sum_per_row = np.asarray(dataMatrix.sum(axis=1)).ravel()

    row_average = np.zeros_like(sum_per_row)
    row_average[nonzero_rows] = sum_per_row[nonzero_rows] / interactions_per_row[nonzero_rows]

    for row_index in range(dataMatrix.shape[0]):
        start_position = dataMatrix.indptr[row_index]
        end_position = dataMatrix.indptr[row_index + 1]

        if end_position > start_position:
            dataMatrix.data[start_position:end_position] -= row_average[row_index]

    return dataMatrix


def _apply_pearson_correlation(dataMatrix):
    dataMatrix = check_matrix(dataMatrix, "csc")

    interactions_per_col = np.ediff1d(dataMatrix.indptr)
    nonzero_cols = interactions_per_col > 0
    sum_per_col = np.asarray(dataMatrix.sum(axis=0)).ravel()

    col_average = np.zeros_like(sum_per_col)
    col_average[nonzero_cols] = sum_per_col[nonzero_cols] / interactions_per_col[nonzero_cols]

    for col_index in range(dataMatrix.shape[1]):
        start_position = dataMatrix.indptr[col_index]
        end_position = dataMatrix.indptr[col_index + 1]

        if end_position > start_position:
            dataMatrix.data[start_position:end_position] -= col_average[col_index]

    return check_matrix(dataMatrix, "csr")
