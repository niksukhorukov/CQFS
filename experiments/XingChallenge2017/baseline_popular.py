from data.DataLoader import XingChallenge2017Loader
from experiments.baseline_popular import baseline_popular

def main():
    
    data_loader = XingChallenge2017Loader()
    ICM_name = 'ICM_all'
    percentages = [5, 20, 40, 60, 80, 95]
    baseline_popular(data_loader, ICM_name, percentages)


if __name__ == "__main__":
    main()
