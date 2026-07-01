#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Benchmark cached KNN similarity matrices together with the current evaluator.

This benchmark compares:
    uncached ItemKNNCF fit + EvaluatorHoldout
    cached ItemKNNCF fit + EvaluatorHoldout
    uncached ItemKNNCF fit + EvaluatorHoldoutFast
    cached ItemKNNCF fit + EvaluatorHoldoutFast
"""

import argparse
import contextlib
import io
import json
import os
import sys
import time

REPOSITORY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if REPOSITORY_ROOT not in sys.path:
    sys.path.insert(0, REPOSITORY_ROOT)

import numpy as np


def _install_numpy_compat_aliases():
    """Keep old recommender/evaluator code runnable on modern NumPy."""
    alias_map = {
        "bool": bool,
        "float": float,
        "int": int,
    }

    for alias_name, alias_value in alias_map.items():
        if alias_name not in np.__dict__:
            setattr(np, alias_name, alias_value)


_install_numpy_compat_aliases()

import scipy.sparse as sps

from recsys.Base.Evaluation.Evaluator import EvaluatorHoldout, EvaluatorHoldoutFast
from recsys.Base.Similarity.SimilarityComputationCache import SimilarityComputationCache
from recsys.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender


PROFILE_CONFIG = {
    "sparse": {
        "n_users": 1000,
        "n_items": 400,
        "train_per_user": 8,
        "test_per_user": 2,
    },
    "dense": {
        "n_users": 800,
        "n_items": 300,
        "train_per_user": 30,
        "test_per_user": 4,
    },
}


def _require_cython_similarity():
    try:
        from recsys.Base.Similarity.Cython.Compute_Similarity_Cython import (  # noqa: F401
            Compute_Similarity_Cached_Cython,
            Compute_Similarity_Cython,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Cython similarity extension is unavailable. Compile it with "
            "`cd recsys && python run_compile_all_cython.py` before benchmarking."
        ) from exc


def _quiet_timed_call(function):
    start_time = time.perf_counter()

    with contextlib.redirect_stdout(io.StringIO()):
        result = function()

    return time.perf_counter() - start_time, result


def _generate_holdout(profile_name, seed):
    profile = PROFILE_CONFIG[profile_name]
    rng = np.random.RandomState(seed)

    n_users = profile["n_users"]
    n_items = profile["n_items"]
    train_per_user = profile["train_per_user"]
    test_per_user = profile["test_per_user"]
    total_per_user = train_per_user + test_per_user

    train_rows = []
    train_cols = []
    test_rows = []
    test_cols = []

    for user_id in range(n_users):
        sampled_items = rng.choice(n_items, size=total_per_user, replace=False)

        train_items = sampled_items[:train_per_user]
        test_items = sampled_items[train_per_user:]

        train_rows.extend([user_id] * train_per_user)
        train_cols.extend(train_items.tolist())

        test_rows.extend([user_id] * test_per_user)
        test_cols.extend(test_items.tolist())

    train_data = rng.uniform(0.1, 5.0, size=len(train_rows)).astype(np.float32)
    test_data = np.ones(len(test_rows), dtype=np.float32)

    URM_train = sps.csr_matrix(
        (train_data, (train_rows, train_cols)),
        shape=(n_users, n_items),
        dtype=np.float32,
    )
    URM_test = sps.csr_matrix(
        (test_data, (test_rows, test_cols)),
        shape=(n_users, n_items),
        dtype=np.float32,
    )

    return URM_train, URM_test


def _build_fit_configs(n_trials, n_items):
    topK_values = [20, 50, 100, 150]
    shrink_values = [0, 5, 25, 100, 250]

    configs = []

    for trial_index in range(n_trials):
        configs.append({
            "topK": min(topK_values[trial_index % len(topK_values)], n_items - 1),
            "shrink": shrink_values[trial_index % len(shrink_values)],
            "similarity": "cosine",
            "normalize": trial_index % 2 == 0,
            "feature_weighting": "none",
        })

    return configs


def _fit_recommender(URM_train, fit_config, similarity_cache=None):
    recommender = ItemKNNCFRecommender(URM_train, verbose=False)
    fit_kwargs = fit_config.copy()

    if similarity_cache is not None:
        fit_kwargs["_similarity_cache"] = similarity_cache

    recommender.fit(**fit_kwargs)
    return recommender


def _evaluate_recommender(evaluator, recommender):
    result_dict, _ = evaluator.evaluateRecommender(recommender)
    return result_dict


def _max_sparse_abs_diff(left_matrix, right_matrix):
    diff_matrix = left_matrix - right_matrix

    if diff_matrix.nnz == 0:
        return 0.0

    return float(np.max(np.abs(diff_matrix.data)))


def _assert_sparse_allclose(left_matrix, right_matrix, rtol, atol):
    if left_matrix.shape != right_matrix.shape:
        raise AssertionError(
            "Similarity shape mismatch: {} != {}".format(left_matrix.shape, right_matrix.shape)
        )

    if not np.allclose(left_matrix.toarray(), right_matrix.toarray(), rtol=rtol, atol=atol):
        max_diff = _max_sparse_abs_diff(left_matrix, right_matrix)
        raise AssertionError("Cached similarity does not match uncached output, max diff {}".format(max_diff))


def _flatten_metrics(result_dict):
    flat_metrics = {}

    for cutoff, cutoff_result_dict in result_dict.items():
        for metric_name, metric_value in cutoff_result_dict.items():
            flat_metrics["{}@{}".format(metric_name, cutoff)] = float(metric_value)

    return flat_metrics


def _max_metric_abs_diff(left_result_dict, right_result_dict, require_same_keys=True):
    left_metrics = _flatten_metrics(left_result_dict)
    right_metrics = _flatten_metrics(right_result_dict)

    if require_same_keys and set(left_metrics.keys()) != set(right_metrics.keys()):
        missing_left = sorted(set(right_metrics.keys()) - set(left_metrics.keys()))
        missing_right = sorted(set(left_metrics.keys()) - set(right_metrics.keys()))
        raise AssertionError(
            "Metric key mismatch. Missing left: {}, missing right: {}".format(missing_left, missing_right)
        )

    if not require_same_keys:
        missing_left = sorted(set(right_metrics.keys()) - set(left_metrics.keys()))

        if len(missing_left) > 0:
            raise AssertionError("Metric key mismatch. Missing left: {}".format(missing_left))

    max_diff = 0.0

    for metric_name in right_metrics:
        metric_diff = abs(left_metrics[metric_name] - right_metrics[metric_name])
        max_diff = max(max_diff, metric_diff)

    return max_diff


def _sum_values(values):
    return float(sum(values))


def _safe_speedup(baseline_seconds, candidate_seconds):
    if candidate_seconds <= 0.0:
        return float("inf")

    return float(baseline_seconds / candidate_seconds)


def _benchmark_profile(profile_name, args):
    URM_train, URM_test = _generate_holdout(profile_name, args.seed)
    full_evaluator = EvaluatorHoldout(URM_test, cutoff_list=[args.cutoff], verbose=False)
    fast_evaluator = EvaluatorHoldoutFast(URM_test, cutoff_list=[args.cutoff], verbose=False)
    fit_configs = _build_fit_configs(args.n_trials, URM_train.shape[1])

    baseline_fit_times = []
    baseline_full_eval_times = []
    baseline_recommenders = []
    baseline_full_results = []

    for fit_config in fit_configs:
        fit_time, recommender = _quiet_timed_call(
            lambda fit_config=fit_config: _fit_recommender(URM_train, fit_config)
        )
        eval_time, result_dict = _quiet_timed_call(
            lambda recommender=recommender: _evaluate_recommender(full_evaluator, recommender)
        )

        baseline_fit_times.append(fit_time)
        baseline_full_eval_times.append(eval_time)
        baseline_recommenders.append(recommender)
        baseline_full_results.append(result_dict)

    baseline_fast_eval_times = []
    baseline_fast_results = []
    max_fast_metric_abs_diff = 0.0

    for trial_index, recommender in enumerate(baseline_recommenders):
        eval_time, result_dict = _quiet_timed_call(
            lambda recommender=recommender: _evaluate_recommender(fast_evaluator, recommender)
        )

        metric_abs_diff = _max_metric_abs_diff(
            baseline_full_results[trial_index],
            result_dict,
            require_same_keys=False,
        )

        if metric_abs_diff > args.metric_atol:
            raise AssertionError(
                "Fast evaluator metrics do not match full evaluator metrics, max diff {}".format(metric_abs_diff)
            )

        max_fast_metric_abs_diff = max(max_fast_metric_abs_diff, metric_abs_diff)
        baseline_fast_eval_times.append(eval_time)
        baseline_fast_results.append(result_dict)

    similarity_cache = SimilarityComputationCache(
        max_memory_bytes=args.cache_memory_mb * 1024 ** 2,
        verbose=False,
    )

    cache_build_time, cached_similarity_data = _quiet_timed_call(
        lambda: similarity_cache.get_similarity_data(URM_train, similarity="cosine")
    )

    if cached_similarity_data is None:
        raise RuntimeError("Similarity cache was rejected by the configured memory budget.")

    cached_fit_times = []
    cached_full_eval_times = []
    cached_fast_eval_times = []
    cached_recommenders = []
    cached_full_results = []
    cached_fast_results = []
    max_similarity_abs_diff = 0.0
    max_cached_full_metric_abs_diff = 0.0
    max_cached_fast_metric_abs_diff = 0.0

    for trial_index, fit_config in enumerate(fit_configs):
        fit_time, recommender = _quiet_timed_call(
            lambda fit_config=fit_config: _fit_recommender(URM_train, fit_config, similarity_cache)
        )
        eval_time, result_dict = _quiet_timed_call(
            lambda recommender=recommender: _evaluate_recommender(full_evaluator, recommender)
        )

        _assert_sparse_allclose(
            baseline_recommenders[trial_index].W_sparse,
            recommender.W_sparse,
            rtol=args.rtol,
            atol=args.atol,
        )
        max_similarity_abs_diff = max(
            max_similarity_abs_diff,
            _max_sparse_abs_diff(baseline_recommenders[trial_index].W_sparse, recommender.W_sparse),
        )

        metric_abs_diff = _max_metric_abs_diff(baseline_full_results[trial_index], result_dict)

        if metric_abs_diff > args.metric_atol:
            raise AssertionError(
                "Cached metrics do not match uncached metrics, max diff {}".format(metric_abs_diff)
            )

        max_cached_full_metric_abs_diff = max(max_cached_full_metric_abs_diff, metric_abs_diff)

        cached_fit_times.append(fit_time)
        cached_full_eval_times.append(eval_time)
        cached_recommenders.append(recommender)
        cached_full_results.append(result_dict)

    for trial_index, recommender in enumerate(cached_recommenders):
        eval_time, result_dict = _quiet_timed_call(
            lambda recommender=recommender: _evaluate_recommender(fast_evaluator, recommender)
        )

        metric_abs_diff = _max_metric_abs_diff(
            cached_full_results[trial_index],
            result_dict,
            require_same_keys=False,
        )

        if metric_abs_diff > args.metric_atol:
            raise AssertionError(
                "Cached fast evaluator metrics do not match cached full evaluator metrics, max diff {}".format(
                    metric_abs_diff
                )
            )

        max_cached_fast_metric_abs_diff = max(max_cached_fast_metric_abs_diff, metric_abs_diff)
        cached_fast_eval_times.append(eval_time)
        cached_fast_results.append(result_dict)

    baseline_fit_total = _sum_values(baseline_fit_times)
    baseline_full_eval_total = _sum_values(baseline_full_eval_times)
    baseline_fast_eval_total = _sum_values(baseline_fast_eval_times)
    initial_full_uncached_total = baseline_fit_total + baseline_full_eval_total
    fast_uncached_total = baseline_fit_total + baseline_fast_eval_total

    cached_fit_total = _sum_values(cached_fit_times)
    cached_full_eval_total = _sum_values(cached_full_eval_times)
    cached_fast_eval_total = _sum_values(cached_fast_eval_times)
    cached_full_total_warm = cached_fit_total + cached_full_eval_total
    cached_full_total_cold = cache_build_time + cached_full_total_warm
    cached_fast_total_warm = cached_fit_total + cached_fast_eval_total
    cached_fast_total_cold = cache_build_time + cached_fast_total_warm

    initial_full_warm_from_second_total = _sum_values(baseline_fit_times[1:]) + _sum_values(baseline_full_eval_times[1:])
    cached_full_warm_from_second_total = _sum_values(cached_fit_times[1:]) + _sum_values(cached_full_eval_times[1:])
    cached_fast_warm_from_second_total = _sum_values(cached_fit_times[1:]) + _sum_values(cached_fast_eval_times[1:])

    return {
        "profile": profile_name,
        "n_trials": args.n_trials,
        "matrix": {
            "n_users": URM_train.shape[0],
            "n_items": URM_train.shape[1],
            "train_nnz": int(URM_train.nnz),
            "test_nnz": int(URM_test.nnz),
            "train_density": float(URM_train.nnz / (URM_train.shape[0] * URM_train.shape[1])),
            "raw_cache_nnz": int(cached_similarity_data.raw_similarity.nnz),
            "raw_cache_memory_mb": float(cached_similarity_data.memory_bytes / 1024 ** 2),
        },
        "seconds": {
            "initial_full_uncached_fit": baseline_fit_total,
            "initial_full_uncached_eval": baseline_full_eval_total,
            "initial_full_uncached_total": initial_full_uncached_total,
            "fast_uncached_fit": baseline_fit_total,
            "fast_uncached_eval": baseline_fast_eval_total,
            "fast_uncached_total": fast_uncached_total,
            "cache_build": float(cache_build_time),
            "cached_full_fit": cached_fit_total,
            "cached_full_eval": cached_full_eval_total,
            "cached_full_total_warm_excluding_build": cached_full_total_warm,
            "cached_full_total_cold_including_build": cached_full_total_cold,
            "cached_fast_fit": cached_fit_total,
            "cached_fast_eval": cached_fast_eval_total,
            "cached_fast_total_warm_excluding_build": cached_fast_total_warm,
            "cached_fast_total_cold_including_build": cached_fast_total_cold,
            "initial_full_warm_from_second_trial": initial_full_warm_from_second_total,
            "cached_full_warm_from_second_trial": cached_full_warm_from_second_total,
            "cached_fast_warm_from_second_trial": cached_fast_warm_from_second_total,
        },
        "speedup": {
            "matrix_cache_only_cold_total": _safe_speedup(initial_full_uncached_total, cached_full_total_cold),
            "matrix_cache_only_warm_total": _safe_speedup(initial_full_uncached_total, cached_full_total_warm),
            "fast_eval_only_total": _safe_speedup(initial_full_uncached_total, fast_uncached_total),
            "combined_cold_total": _safe_speedup(initial_full_uncached_total, cached_fast_total_cold),
            "combined_warm_total": _safe_speedup(initial_full_uncached_total, cached_fast_total_warm),
            "combined_warm_from_second_trial": _safe_speedup(
                initial_full_warm_from_second_total,
                cached_fast_warm_from_second_total,
            ),
            "fit_only_cold_including_cache_build": _safe_speedup(baseline_fit_total, cache_build_time + cached_fit_total),
            "fit_only_warm_excluding_cache_build": _safe_speedup(baseline_fit_total, cached_fit_total),
            "fast_eval_only_eval": _safe_speedup(baseline_full_eval_total, baseline_fast_eval_total),
            "cached_fast_eval_only_eval": _safe_speedup(cached_full_eval_total, cached_fast_eval_total),
        },
        "correctness": {
            "max_similarity_abs_diff": max_similarity_abs_diff,
            "max_fast_metric_abs_diff": max_fast_metric_abs_diff,
            "max_cached_full_metric_abs_diff": max_cached_full_metric_abs_diff,
            "max_cached_fast_metric_abs_diff": max_cached_fast_metric_abs_diff,
            "similarity_rtol": args.rtol,
            "similarity_atol": args.atol,
            "metric_atol": args.metric_atol,
        },
        "per_trial_seconds": {
            "initial_full_uncached_fit": baseline_fit_times,
            "initial_full_uncached_eval": baseline_full_eval_times,
            "fast_uncached_eval": baseline_fast_eval_times,
            "cached_fit": cached_fit_times,
            "cached_full_eval": cached_full_eval_times,
            "cached_fast_eval": cached_fast_eval_times,
        },
    }


def _print_result_table(results):
    header = (
        "profile",
        "trials",
        "initial",
        "cache full",
        "fast only",
        "cache+fast",
        "cache x",
        "fast x",
        "combo x",
    )
    row_format = "{:<8} {:>6} {:>11} {:>11} {:>11} {:>11} {:>8} {:>8} {:>8}"

    print(row_format.format(*header))

    for result in results:
        seconds = result["seconds"]
        speedup = result["speedup"]

        print(row_format.format(
            result["profile"],
            result["n_trials"],
            "{:.4f}s".format(seconds["initial_full_uncached_total"]),
            "{:.4f}s".format(seconds["cached_full_total_cold_including_build"]),
            "{:.4f}s".format(seconds["fast_uncached_total"]),
            "{:.4f}s".format(seconds["cached_fast_total_cold_including_build"]),
            "{:.2f}x".format(speedup["matrix_cache_only_cold_total"]),
            "{:.2f}x".format(speedup["fast_eval_only_total"]),
            "{:.2f}x".format(speedup["combined_cold_total"]),
        ))


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark cached KNN similarity matrices with the current evaluator."
    )
    parser.add_argument(
        "--profile",
        choices=["sparse", "dense", "both"],
        default="both",
        help="Synthetic matrix profile to benchmark.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=30,
        help="Number of KNN fit/eval configurations per profile.",
    )
    parser.add_argument(
        "--cutoff",
        type=int,
        default=10,
        help="Evaluator cutoff.",
    )
    parser.add_argument(
        "--evaluator",
        choices=["both"],
        default="both",
        help="Run full and fast evaluator scenarios.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed.",
    )
    parser.add_argument(
        "--cache-memory-mb",
        type=int,
        default=2048,
        help="Similarity cache memory budget in MB.",
    )
    parser.add_argument(
        "--output",
        default="results/benchmark_cached_similarity_eval.json",
        help="JSON output path. Use an empty string to skip writing JSON.",
    )
    parser.add_argument("--rtol", type=float, default=1e-5, help="Similarity allclose relative tolerance.")
    parser.add_argument("--atol", type=float, default=1e-6, help="Similarity allclose absolute tolerance.")
    parser.add_argument("--metric-atol", type=float, default=1e-10, help="Metric absolute tolerance.")

    args = parser.parse_args()

    if args.n_trials < 1:
        raise ValueError("--n-trials must be >= 1")

    if args.cutoff < 1:
        raise ValueError("--cutoff must be >= 1")

    return args


def main():
    args = _parse_args()
    _require_cython_similarity()

    if args.profile == "both":
        profiles = ["sparse", "dense"]
    else:
        profiles = [args.profile]

    results = []
    start_time = time.perf_counter()

    for profile_index, profile_name in enumerate(profiles):
        profile_args = argparse.Namespace(**vars(args))
        profile_args.seed = args.seed + profile_index
        results.append(_benchmark_profile(profile_name, profile_args))

    output_payload = {
        "benchmark": "cached_similarity_with_full_and_fast_evaluators",
        "evaluator": "EvaluatorHoldout baseline and EvaluatorHoldoutFast validation evaluator",
        "python": sys.version,
        "profiles": profiles,
        "total_wall_time_sec": float(time.perf_counter() - start_time),
        "results": results,
    }

    _print_result_table(results)

    if args.output:
        output_dir = os.path.dirname(args.output)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(args.output, "w") as output_file:
            json.dump(output_payload, output_file, indent=2, sort_keys=True)

        print("Wrote {}".format(args.output))


if __name__ == "__main__":
    main()
