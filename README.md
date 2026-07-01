# Collaborative-driven Quantum Feature Selection

This repository was developed by Riccardo Nembrini, PhD student at Politecnico di Milano.
See the websites of our [quantum computing group](https://quantum.polimi.it/) and of our
[recommender systems group](http://recsys.deib.polimi.it/) for more information on our teams and works.
This repository contains the source code for the article "**Feature Selection for Recommender Systems with Quantum
Computing**".

Here we explain how to install dependencies, setup the connection to D-Wave Leap quantum cloud services and how to run
experiments included in this repository.

## Installation

> NOTE: This repository requires Python 3.7

It is suggested to install all the required packages into a new Python environment. So, after repository checkout, enter
the repository folder and run the following commands to create a new environment:

If you're using `virtualenv`:

```bash
virtualenv -p python3 cqfs
source cqfs/bin/activate
```

If you're using `conda`:

```bash
conda create -n cqfs python=3.7 anaconda
conda activate cqfs
```

>Remember to add this project in the PYTHONPATH environmental variable if you plan to run the experiments 
on the terminal:
>```bash
>export PYTHONPATH=$PYTHONPATH:/path/to/project/folder
>```

Then, make sure you correctly activated the environment and install all the required packages through `pip`:

```bash
pip install -r requirements.txt
```

After installing the dependencies, it is suggested to compile Cython code in the repository.

In order to compile you must first have installed: `gcc` and `python3 dev`. Under Linux those can be installed with the
following commands:

```bash
sudo apt install gcc 
sudo apt-get install python3-dev
```

If you are using Windows as operating system, the installation procedure is a bit more complex. You may refer
to [THIS](https://github.com/cython/cython/wiki/InstallingOnWindows) guide.

Now you can compile all Cython algorithms by running the following command. The script will compile within the current
active environment. The code has been developed for Linux and Windows platforms. During the compilation you may see some
warnings.

```bash
python recsys/run_compile_all_cython.py
```

## D-Wave Setup

In order to make use of D-Wave cloud services you must first sign-up to [D-Wave Leap](https://cloud.dwavesys.com/leap/)
and get your API token.

Then, you need to run the following command in the newly created Python environment:

```bash
dwave setup
```

This is a guided setup for D-Wave Ocean SDK. When asked to select non-open-source packages to install you should
answer `y` and install at least _D-Wave Drivers_ (the D-Wave Problem Inspector package is not required, but could be
useful to analyse problem solutions, if solving problems with the QPU only).

Then, continue the configuration by setting custom properties (or keeping the default ones, as we suggest), apart from
the `Authentication token` field, where you should paste your API token obtained on the D-Wave Leap dashboard.

You should now be able to connect to D-Wave cloud services. In order to verify the connection, you can use the following
command, which will send a test problem to D-Wave's QPU:

```bash
dwave ping
```

## Running CQFS Experiments

Run the following commands from the repository root unless stated otherwise.

### 1. Prepare Datasets

First, place the original dataset archives in the expected offline-data directories:

| Dataset | Expected file |
| --- | --- |
| The Movies Dataset | `recsys/Data_manager_offline_datasets/TheMoviesDataset/the-movies-dataset.zip` |
| CiteULike_a | `recsys/Data_manager_offline_datasets/CiteULike/CiteULike_a_t.zip` |
| Xing Challenge 2017 | `recsys/Data_manager_offline_datasets/XingChallenge2017/xing_challenge_data_2017.zip` |

For The Movies Dataset, download
[The Movies Dataset from Kaggle](https://www.kaggle.com/rounakbanik/the-movies-dataset). For CiteULike_a, download
[this archive](https://polimi365-my.sharepoint.com/:u:/g/personal/10322330_polimi_it/EcjHpkI8TQdHnFVwVMkNGN4BmNkurMWw79sU8kpt4wk8eA?e=QYhdbz).
Xing Challenge 2017 data cannot be redistributed here.

Generate the preprocessed splits with the matching split script:

```bash
python data/split_CiteULike_a.py
python data/split_TheMoviesDataset.py
python data/split_XingChallenge2017.py
```

The split scripts write under `recsys/Data_manager_split_datasets/`.

### 2. Compile Cython

Many recommenders and the cached similarity scorer require compiled Cython extensions. Recompile after changing Cython
files or after creating a fresh environment:

```bash
python recsys/run_compile_all_cython.py
```

### 3. Runtime Performance Flags

The experiment scripts expose the same validation and KNN-cache controls:

```bash
--fast-validation-evaluator
--no-fast-validation-evaluator
--similarity-cache
--no-similarity-cache
--similarity-cache-memory-mb 2048
```

Defaults:

- `--fast-validation-evaluator` is enabled by default for validation/search.
- `--similarity-cache` is enabled by default for KNN content/collaborative searches.
- Test/final evaluation still uses the original full `EvaluatorHoldout`.
- `baseline_CFW.py` exposes only the fast-evaluator flag because it does not use the KNN similarity-cache path.
- The cache memory budget is per process. Parallel runs can use roughly `process_count * similarity_cache_memory_mb`.

Use `--help` on any dataset runner to see the exact options:

```bash
python experiments/CiteULike_a/CollaborativeFiltering.py --help
python experiments/CiteULike_a/CQFSTrainer.py --help
python experiments/CiteULike_a/baseline_CFW.py --help
```

For very large matrices, for example `100000 x 100000`, start with an explicit memory budget and disable cache if the
raw co-occurrence matrix is too large for the machine:

```bash
python experiments/CiteULike_a/CollaborativeFiltering.py --similarity-cache-memory-mb 8192
python experiments/CiteULike_a/CollaborativeFiltering.py --no-similarity-cache
```

If the cache budget is exceeded, the code falls back to the normal Cython similarity computation for that matrix.

### 4. Recommended Experiment Order

Each dataset has its own scripts under `experiments/CiteULike_a/`, `experiments/TheMoviesDataset/`, and
`experiments/XingChallenge2017/`. Replace `CiteULike_a` in the examples with another dataset folder as needed.

First optimize collaborative recommenders. CQFS trainers load these saved CF hyperparameters:

```bash
python experiments/CiteULike_a/CollaborativeFiltering.py
```

Run the baselines:

```bash
# ItemKNN content-based with all features
python experiments/CiteULike_a/baseline_CBF.py

# ItemKNN content-based with TF-IDF feature selection
python experiments/CiteULike_a/baseline_TFIDF.py

# CFeCBF feature-weighting baseline
python experiments/CiteULike_a/baseline_CFW.py
```

Run CQFS selection, then tune recommenders on the selected features:

```bash
python experiments/CiteULike_a/CQFS.py
python experiments/CiteULike_a/CQFSTrainer.py
```

Run CQFSTT selection and training:

```bash
python experiments/CiteULike_a/CQFSTT.py
python experiments/CiteULike_a/CQFSTTTrainer.py
```

Results are written under `results/`, grouped by dataset, ICM, recommender, and experiment id.

### 5. Common Command Variants

Use the default optimized validation/cache path:

```bash
python experiments/TheMoviesDataset/baseline_CBF.py
```

Compare against the original validation evaluator and uncached similarity computation:

```bash
python experiments/TheMoviesDataset/baseline_CBF.py --no-fast-validation-evaluator --no-similarity-cache
```

Use fast validation but disable matrix cache for a large or memory-constrained run:

```bash
python experiments/XingChallenge2017/CollaborativeFiltering.py --no-similarity-cache
```

Increase cache budget for repeated KNN searches on a matrix that fits in memory:

```bash
python experiments/CiteULike_a/CQFSTrainer.py --similarity-cache-memory-mb 8192
```

### 6. Benchmark And Correctness Checks

Run focused correctness tests for the new evaluator and similarity cache:

```bash
python -m unittest recsys/Base/Evaluation/Evaluator_fast_test.py
python -m unittest recsys/Base/Similarity/Compute_similarity_cache_test.py
```

Run the synthetic benchmark comparing original, cached, fast evaluator, and cached+fast paths:

```bash
python experiments/benchmark_cached_similarity_eval.py --profile both --n-trials 30 --cache-memory-mb 2048
```

For a quick smoke benchmark:

```bash
python experiments/benchmark_cached_similarity_eval.py --profile sparse --n-trials 3 --output ""
```

### 7. D-Wave Runtime Notes

Each selection with D-Wave Leap hybrid service takes roughly 8 seconds for The Movies Dataset and roughly 30 seconds for
CiteULike_a. Running the scripts with all default CQFS hyperparameters can consume most or all free D-Wave Leap time and
may cause errors or invalid selections after the quota is exhausted.

For D-Wave runs, consider reducing the CQFS hyperparameter grids in the dataset-specific scripts or running a single
collaborative model first. This does not apply to local simulated annealing, which runs locally. Xing Challenge 2017
experiments run directly on the D-Wave QPU, so be careful when increasing sampler reads or annealing time.

## Acknowledgements
Software produced by Riccardo Nembrini.
Recommender systems library by Maurizio Ferrari Dacrema.

Article authors: Riccardo Nembrini, Maurizio Ferrari Dacrema, Paolo Cremonesi
