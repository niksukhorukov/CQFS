from data.DataLoader import TheMoviesDatasetLoader
from experiments.baseline_random import baseline_random

def main():
    percentages = [5, 10, 20, 30, 40, 60, 80, 95]
    data_loader = TheMoviesDatasetLoader()
    ICM_name = 'ICM_metadata'
    for seed in range(1):
        baseline_random(data_loader, ICM_name, percentages, seed)


if __name__ == "__main__":
    main()
