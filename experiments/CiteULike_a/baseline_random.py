from data.DataLoader import CiteULike_aLoader
from experiments.baseline_random import baseline_random

def main():
    data_loader = CiteULike_aLoader()
    ICM_name = 'ICM_title_abstract'
    percentages = [20, 30, 40, 60, 80, 95]
    for seed in range(1):
        baseline_random(data_loader, ICM_name, percentages, seed)


if __name__ == "__main__":
    main()
