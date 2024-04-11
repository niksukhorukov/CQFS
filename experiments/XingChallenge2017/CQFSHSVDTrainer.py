from core.CQFSHSVDSampler import CQFSHSVDSampler
from data.DataLoader import XingChallenge2017Loader
from experiments.train_CQFSHSVD import train_CQFSHSVD
from recsys.Recommender_import_list import (
    ItemKNNCFRecommender, PureSVDItemRecommender, RP3betaRecommender,
)

import ctypes
def mkl_set_num_threads(num_threads=20):
    def mkl_set_num_threads(cores):
        mkl_rt.mkl_set_num_threads(ctypes.byref(ctypes.c_int(cores)))
        
    mkl_rt = ctypes.CDLL('libmkl_rt.so')
    mkl_max_threads = mkl_rt.mkl_get_max_threads()
    mkl_set_num_threads(num_threads)
    print(f'[mkl]: set up num_threads={mkl_rt.mkl_get_max_threads()}/{mkl_max_threads}')



def main():
    num_threads = 20
    mkl_set_num_threads(num_threads)
    data_loader = XingChallenge2017Loader()
    ICM_name = 'ICM_all'

    parameter_product = True
    parameter_per_recommender = False
    percentages = [40, 60, 80, 95]
    alphas = [0.4, 0.7, 0.9]
    ranks = [100, 200]
    degs = [-1.5, -0.5, 0.5]

    CF_recommender_classes = [
        PureSVDItemRecommender,
    ]
    sampler = CQFSHSVDSampler()

    cpu_count_div = 1
    cpu_count_sub = 0

    train_CQFSHSVD(
        data_loader=data_loader, ICM_name=ICM_name,
        percentages=percentages, alphas=alphas, ranks=ranks,
        degs=degs,
        CF_recommender_classes=CF_recommender_classes,
        cpu_count_div=cpu_count_div, cpu_count_sub=cpu_count_sub,
        sampler=sampler,
        parameter_product=parameter_product,
        parameter_per_recommender=parameter_per_recommender,
    )


if __name__ == '__main__':
    main()
