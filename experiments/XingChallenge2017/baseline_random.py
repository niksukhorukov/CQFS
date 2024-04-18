from data.DataLoader import XingChallenge2017Loader
from experiments.baseline_random import baseline_random

def main():
    percentages = [5, 20, 40, 60, 80, 95]
    data_loader = XingChallenge2017Loader()
    ICM_name = 'ICM_all'
    for seed in range(1):
        baseline_random(data_loader, ICM_name, percentages, seed)


if __name__ == "__main__":
    main()
