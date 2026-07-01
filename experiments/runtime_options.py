import argparse

from recsys.Base.Evaluation.Evaluator import EvaluatorHoldout, EvaluatorHoldoutFast


DEFAULT_SIMILARITY_CACHE_MEMORY_MB = 2048


def add_runtime_options(parser, include_similarity_cache=True):
    parser.add_argument(
        "--fast-validation-evaluator",
        dest="use_fast_validation_evaluator",
        action="store_true",
        default=True,
        help="Use EvaluatorHoldoutFast for validation/search. Enabled by default.",
    )
    parser.add_argument(
        "--no-fast-validation-evaluator",
        dest="use_fast_validation_evaluator",
        action="store_false",
        help="Use the original EvaluatorHoldout for validation/search.",
    )

    if include_similarity_cache:
        parser.add_argument(
            "--similarity-cache",
            dest="enable_similarity_cache",
            action="store_true",
            default=True,
            help="Use cached raw KNN co-occurrence matrices when the memory budget allows it. Enabled by default.",
        )
        parser.add_argument(
            "--no-similarity-cache",
            dest="enable_similarity_cache",
            action="store_false",
            help="Disable cached raw KNN co-occurrence matrices.",
        )
        parser.add_argument(
            "--similarity-cache-memory-mb",
            type=int,
            default=DEFAULT_SIMILARITY_CACHE_MEMORY_MB,
            help="Per-process similarity cache memory budget in MB. Default: %(default)s.",
        )

    return parser


def parse_runtime_options(description, include_similarity_cache=True):
    parser = argparse.ArgumentParser(description=description)
    add_runtime_options(parser, include_similarity_cache=include_similarity_cache)
    args = parser.parse_args()

    if include_similarity_cache and args.similarity_cache_memory_mb <= 0:
        parser.error("--similarity-cache-memory-mb must be greater than zero")

    return args


def runtime_kwargs(args, include_similarity_cache=True):
    kwargs = {
        "use_fast_validation_evaluator": args.use_fast_validation_evaluator,
    }

    if include_similarity_cache:
        kwargs["enable_similarity_cache"] = args.enable_similarity_cache
        kwargs["similarity_cache_memory_mb"] = args.similarity_cache_memory_mb

    return kwargs


def make_validation_evaluator(URM_validation, cutoff_list, ignore_items=None, use_fast_validation_evaluator=True):
    evaluator_class = EvaluatorHoldoutFast if use_fast_validation_evaluator else EvaluatorHoldout
    return evaluator_class(URM_validation, cutoff_list=cutoff_list, ignore_items=ignore_items)


def print_runtime_options(args, include_similarity_cache=True):
    print("Runtime options:")
    print("  fast_validation_evaluator={}".format("on" if args.use_fast_validation_evaluator else "off"))

    if include_similarity_cache:
        print("  similarity_cache={}".format("on" if args.enable_similarity_cache else "off"))
        print("  similarity_cache_memory_mb={}".format(args.similarity_cache_memory_mb))
        print("  note: similarity cache memory budget is per process")
