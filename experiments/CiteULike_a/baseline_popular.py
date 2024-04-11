from data.DataLoader import CiteULike_aLoader
from experiments.baseline_popular import baseline_popular

def main():
    data_loader = CiteULike_aLoader()
    ICM_name = 'ICM_title_abstract'
    percentages = [5, 20, 30, 40, 60, 80, 95]
    baseline_popular(data_loader, ICM_name, percentages)


if __name__ == "__main__":
    main()
