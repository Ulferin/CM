import numpy as np
import itertools
from multiprocessing import Pool

from sklearn.model_selection import ParameterGrid

class GridSearch():

    def gridsearch(self, par_grid):
        est_pars = par_grid.pop("est_pars")
        train_pars = par_grid.pop("train_pars")

        input_units = train_pars['training_data'][0].shape[1]
        output_units = train_pars['training_data'][1].shape[1] if len(train_pars['training_data'][1].shape) == 2 else 1
        est_pars['sizes'].insert(0, input_units)
        est_pars['sizes'].append(output_units)


        est = est_pars.pop('estimator')
        net = est(**est_pars)

        net.train(**train_pars)
        return net.best_score()

    def fit(self, grid, dataset, test):

        est_pars = list(ParameterGrid(grid.pop('est_pars')))
        train_pars = grid.pop('train_pars')

        pars = [{'est_pars':ep, 'train_pars':train_pars} for ep in est_pars]


        score_file = open(f"src/NN/res/scores/{dataset}_{test}.txt", 'a')
        search_size = len(pars)
        print(f"Executing search over {search_size} configurations.")

        for par_grid in pars:
            est_pars = par_grid.pop("est_pars")
            train_pars = par_grid.pop("train_pars")

            input_units = train_pars['training_data'][0].shape[1]
            output_units = train_pars['training_data'][1].shape[1] if len(train_pars['training_data'][1].shape) == 2 else 1
            est_pars['sizes'].insert(0, input_units)
            est_pars['sizes'].append(output_units)


            est = est_pars.pop('estimator')
            net = est(**est_pars)

            net.train(**train_pars)
            res = net.best_score()
            print(f"Remaining: {search_size}")
            print(res)
            score_file.write(res)

        # with Pool(processes=None) as pool:
        #         for res in (pool.imap_unordered(self.gridsearch, pars)):
        #             search_size -= 1
        #             print(f"Remaining: {search_size}")
        #             print(res)
        #             score_file.write(res)

        score_file.close()