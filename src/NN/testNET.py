import numpy as np
import sys

from src.NN.Network import NR, NC
import src.NN.utils as utils
from src.NN.GridSearch import GridSearch

from multiprocessing import Process, Pool

datasets = {
    'cup': 'data/ML-CUP20-TR.csv',
    'monk1': 'data/monks-1',
    'monk2': 'data/monks-2', 
    'monk3': 'data/monks-3', 
}


if __name__ == '__main__':
    test = sys.argv[1]
    dataset = sys.argv[2]
    grid = len(sys.argv) > 3 and sys.argv[3] == 'grid'

    if dataset == 'cup':
        X_train, X_test, y_train, y_test = utils.load_CUP(datasets[dataset])
    else:
        X_train, X_test, y_train, y_test = utils.load_monk(datasets[dataset])
        X_train, X_test = utils.prepare_data(X_train, X_test)

    # Loads the input and output layers shape
    input_units = X_train.shape[1]
    output_units = y_train.shape[1] if len(y_train.shape) == 2 else 1


    # Performs gridsearch over the specified hyperparameters
    if grid:

        grids = {
            'est_pars': {
                'sizes': [[16, 32], [50, 50], [71, 69]],
                'seed': [0],
                'activation': ['Lrelu'],
                'lmbda': [0, 0.001, 0.01, 0.1],
                'momentum': [0, 0.5, 0.9],
                'debug': [False],
                'epochs': [500],
                'batch_size': [None],
                'eta':[0.0001, 0.001, 0.01],
                'optimizer': [test],
                'eps': [1e-3]
            },
            'train_pars': {
                'training_data': (X_train, y_train),
                'test_data': (X_test, y_test)
            }
    }

        if dataset == 'cup':
            grids['est_pars']['estimator'] = [NR]
        else:
            grids['est_pars']['estimator'] = [NC]
        
        gs = GridSearch()
        gs.fit(grids, dataset, test)
    
    else:

        # TODO: aggiungere json di configurazione
        params = {
            'cup': {
                'SGD': {
                    'sizes': [input_units, 16, 32, output_units],
                    'activation': 'Lrelu',
                    'seed': 0,
                    'lmbda': 0.,
                    'momentum': 0.,
                    'epochs': 5000,
                    'batch_size': None,
                    'eta': 0.0001,
                    'optimizer': test,
                    'debug': True,
                    'eps':1e-3
                },
                'SGM': {
                    'sizes': [input_units, 16, 32, output_units],
                    'activation': 'Lrelu',
                    'seed': 0,
                    'lmbda': 0.,
                    'momentum': 0.,
                    'epochs': 500,
                    'batch_size': None,
                    'eta': 0.0001,
                    'optimizer': test,
                    'debug': True,
                    'eps':1e-3
                }
            },
            'monk': {
                'SGD': {
                    'sizes': [input_units, 3, output_units],
                    'activation': 'Lrelu',
                    'seed': 0,
                    'lmbda': 0.001,
                    'momentum': 0.5,
                    'epochs': 50000,
                    'batch_size': None,
                    'eta': 0.05,
                    'optimizer': test,
                    'debug': True,
                    'eps':1e-3
                },
                'SGM': {
                    'sizes': [input_units, 3, output_units],
                    'activation': 'Lrelu',
                    'seed': 0,
                    'lmbda': 0.001,
                    'momentum': 0.0,
                    'epochs': 50000,
                    'batch_size': None,
                    'eta': 0.1,
                    'optimizer': test,
                    'debug': True,
                    'eps':1e-3
                }
            }
        }

        if dataset == 'cup':
            net = NR(**params[dataset][test])
        else:
            dataset = 'monk'
            net = NC(**params[dataset][test])

        net.train((X_train, y_train), test_data=(X_test, y_test))
        print(net.best_score())
        # net.plot_grad('gradient')

    