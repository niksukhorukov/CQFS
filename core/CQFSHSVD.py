import re
import time

import numpy as np

from recsys.Base.DataIO import DataIO
from utils.naming import get_experiment_id
from utils.statistics import similarity_statistics, BQM_statistics
from scipy.linalg import cholesky
from scipy.sparse.linalg import svds
from scipy.sparse import diags

from core.maxvol import maxvol_rect

class CQFSHSVD:
    SAVED_MODELS_FILE = 'saved_CQFSHSVD_models.zip'
    STATISTICS_FILE = 'statistics'
    TIMINGS_FILE = 'timings'

    def __init__(self, ICM_train, URM_train, base_folder_path, *, sampler):

        self.n_items, self.n_features = ICM_train.shape
        self.ICM_train = ICM_train.copy()
        self.URM_train = URM_train.copy()

        if re.match('.*/.*ICM.*/.*Recommender.*/', base_folder_path) is None:
            self.__print("[WARNING] base_folder_path has a custom format, we suggest to use the following one for "
                         "compatibility with other classes:\n"
                         "DatasetName/ICMName/CFRecommenderName/")

        self.base_folder_path = base_folder_path if base_folder_path[-1] == '/' else f"{base_folder_path}/"
        self.dataIO = DataIO(self.base_folder_path)

        self.statistics = {}

        self.timings = self.__load_timings()

        ##################################################
        # Model variables

        # self.IPMs = {}
        self.FPMs = {}
        # self.BQMs = {}
        self.selections = {}

        ##################################################

        ##################################################
        # Solver initialization

        self.solver = sampler
        self.selection_type = 'cqfs_hsvd'
        self.timings['avg_response_time'] = {}
        self.timings['n_select_experiments'] = {}

        if self.timings['avg_response_time'].get(self.selection_type) is None:
            self.timings['avg_response_time'][self.selection_type] = 0
            self.timings['n_select_experiments'][self.selection_type] = 0

    @staticmethod
    def __print(msg):
        print(f"CQFSHSVD: {msg}")

    def __save_statistics(self):
        self.dataIO.save_data(self.STATISTICS_FILE, self.statistics)

    def __save_timings(self):
        self.dataIO.save_data(self.TIMINGS_FILE, self.timings)

    def __load_timings(self):
        timings = {}
        try:
            timings = self.dataIO.load_data(self.TIMINGS_FILE)
        except FileNotFoundError:
            self.__print("No timings file found.")
        return timings

    def __load_base_model(self, model):

        model_file = f'{model}.zip'

        try:
            if model == 'FPM_K':
                self.FPM_K = self.dataIO.load_data(model_file)['FPM_K']
            elif model == 'FPM_E':
                self.FPM_E = self.dataIO.load_data(model_file)['FPM_E']
            return True

        except FileNotFoundError:
            return False

    def __load_previously_saved_models(self):

        self.__print("Trying to load previously saved models.")

        saved_models = {
            'FPM_K': False,
            'FPM_E': False,
        }

        try:
            saved_models = self.dataIO.load_data(self.SAVED_MODELS_FILE)

            for model in saved_models:
                if saved_models[model]:
                    saved_models[model] = self.__load_base_model(model)

        except FileNotFoundError:
            self.__print("No model saved for this set of experiments.")

        self.dataIO.save_data(self.SAVED_MODELS_FILE, saved_models)
        return saved_models

    @staticmethod
    def __p_to_k(p, n_features):
        assert p is not None, "Please, choose a selection percentage." \
                              "The value should be between 0 and 1 or between 0 and 100."

        if 1 < p <= 100:
            p /= 100
        elif p > 100 or p < 0:
            raise ValueError("Percentage value should be between 0 and 1 or between 0 and 100.")

        return n_features * p


    def __get_selection_from_sample(self, sample):

        selection = np.zeros(self.n_features, dtype=bool)
        for k, v in sample.items():
            if v == 1:
                ind = int(k)
                selection[ind] = True

        return selection

    def __get_experiment_dataIO(self, expID):
        experiment_folder_path = f"{self.base_folder_path}{expID}/"
        return DataIO(experiment_folder_path)

    def __get_selection_dataIO(self, expID):
        selection_folder_path = f"{self.base_folder_path}{expID}/{self.selection_type}/"
        return DataIO(selection_folder_path)

    def __save_selection(self, expID, selection):

        selection_dataIO = self.__get_selection_dataIO(expID)
        selection_dict = {
            'selection': selection,
        }
        selection_dataIO.save_data(self.selection_type, selection_dict)
        self.__print(f"[{expID}] Selection saved.")

    def __load_selection(self, expID):

        selection_dataIO = self.__get_selection_dataIO(expID)

        try:
            selection = selection_dataIO.load_data(self.selection_type)['selection']
            self.__print(f"[{expID}] Found an existing selection.")
            self.selections[expID] = selection.copy()

        except FileNotFoundError:
            self.__print(f"[{expID}] No previous selection found.")

    def select(self, alpha, beta, combination_strength=1):
        raise NotImplementedError("Method not implemented yet.")

    def select_p(self, ps, alpha_inv, rank_inv, deg_inv, vartype='BINARY', save_FPM=False):


        # self.__print(f"[{expID}] Starting selection.")

        # self.__print(f"[{expID}] Sampling the problem.")
        response_time = time.time()

        max_k = int(self.__p_to_k(max(ps), self.n_features))

        popularity = np.array(self.URM_train.sum(axis=0)).squeeze()
        popularity[popularity == 0.0] = 1.0
        popularity = popularity ** deg_inv
        sim = self.URM_train @ diags(popularity)
        
        S_collab = (sim.T.dot(sim)).A
        S_collab = S_collab / np.max(S_collab)
        S_collab = S_collab - np.diag(np.diag(S_collab)) + np.eye(S_collab.shape[0])
        S_maxvol = (1 - alpha_inv) * np.eye(S_collab.shape[0]) + alpha_inv * S_collab
        t4_time = time.time() - response_time
        print('S preparation', t4_time)
        print('S shape', S_maxvol.shape)
        response_time1 = time.time()
        L_S_maxvol = np.linalg.cholesky(S_maxvol)
        ls_time = time.time() - response_time1
        print('cholesky time', ls_time)
        response_time1 = time.time()
        item_saturated = L_S_maxvol.T @ self.ICM_train
        saturated_time = time.time() - response_time1
        print('saturation time', saturated_time)
        response_time1 = time.time()
        _, S_item, Vt_item = svds(item_saturated, k=rank_inv)
        svd_time = time.time() - response_time1
        print('svd time', svd_time)
        response_time1 = time.time()
        order = np.argsort(S_item)[::-1]
        Q = np.ascontiguousarray(Vt_item[order, :].T)
        print('Q shape', Q.shape)

        pivots, _ = maxvol_rect(Q, dr_min=max_k-Q.shape[1], dr_max=max_k-Q.shape[1])
        pivot_time = time.time() - response_time1
        print('maxvol time', pivot_time)
        for p in ps:
            k = int(self.__p_to_k(p, self.n_features))

            expID = get_experiment_id(alpha_inv, rank_inv, deg_inv, p=p)
            self.__load_selection(expID)
            # if self.selections.get(expID) is not None:
            #     return self.selections[expID].copy()
            features = pivots[:k]
            best_sample = {i:0 for i in range(self.ICM_train.shape[1])}
            for feature in features:
                best_sample[feature] = 1
            
            response_time = time.time() - response_time

            experiment_timings = {
                'response_time': response_time,
            }
            selection_dataIO = self.__get_selection_dataIO(expID)
            selection_dataIO.save_data(self.TIMINGS_FILE, experiment_timings)

            n_experiments = self.timings['n_select_experiments'][self.selection_type]
            total_response_time = self.timings['avg_response_time'][self.selection_type] * n_experiments
            n_experiments += 1
            self.timings['n_select_experiments'][self.selection_type] = n_experiments
            self.timings['avg_response_time'][self.selection_type] = (total_response_time + response_time) / n_experiments
            self.__save_timings()

            selection = self.__get_selection_from_sample(best_sample)

            self.__print(f"[{expID}] Selected {selection.sum()} features in {response_time} sec.")

            self.selections[expID] = selection.copy()
            self.__save_selection(expID, selection)
        # return selection

    def select_many_p(self, ps, alphas, ranks, degs, vartype='BINARY', save_FPMs=False, parameter_product=True):
        if parameter_product:
            for alpha_inv in alphas:
                for rank_inv in ranks:
                    for deg_inv in degs:
                        self.select_p(ps, alpha_inv, rank_inv, deg_inv, vartype, save_FPMs)
        else:
            args_zip = zip(ps, alphas, ranks, degs)
            for args in args_zip:
                self.select_p(args[0], args[1], args[2], args[3], vartype, save_FPMs)
