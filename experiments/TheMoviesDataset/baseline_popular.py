from data.DataLoader import TheMoviesDatasetLoader
from experiments.baseline_popular import baseline_popular

def main():
    
    data_loader = TheMoviesDatasetLoader()
    ICM_name = 'ICM_metadata'
    percentages = [5, 10, 20, 30, 40, 60, 80, 95]
    baseline_popular(data_loader, ICM_name, percentages)


if __name__ == "__main__":
    main()
