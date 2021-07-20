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
                'epochs': [500, 1000],
                'batch_size': [None],
                'eta':[0.0001, 0.001, 0.01, 0.1],
                'optimizer': [test],
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
                    'h1': 79,
                    'h2': 61,
                    'activation': 'Lrelu',
                    'lmbda': 0.01,
                    'momentum': 0.5,
                    'epochs': 1000,
                    'batch_size': None,
                    'eta': 0.001,
                },
                'SGM': {
                    'h1': 79,
                    'h2': 61,
                    'activation': 'Lrelu',
                    'lmbda': 0.,
                    'momentum': 0.7,
                    'epochs': 1000,
                    'batch_size': None,
                    'eta': 0.01
                }
            },
            'monk': {
                'SGD': {
                    'h1': 3,
                    'h2': None,
                    'activation': 'Lrelu',
                    'lmbda': 0.001,
                    'momentum': 0.5,
                    'epochs': 50000,
                    'batch_size': None,
                    'eta': 0.05
                },
                'SGM': {
                    "h1": 3,
                    "h2": 0,
                    "activation": "Lrelu",
                    "lmbda": 0.0,
                    "momentum": 0.0,
                    "epochs": 50000,
                    "batch_size": None,
                    "eta": 0.1
                }
            }
        }

        if dataset == 'cup':
            net = NR([input_units, params[dataset][test]['h1'], params[dataset][test]['h2'], output_units],
                        test,
                        0,
                        params[dataset][test]['epochs'],
                        params[dataset][test]['eta'],
                        params[dataset][test]['activation'],
                        lmbda=params[dataset][test]['lmbda'],
                        momentum=params[dataset][test]['momentum'],
                        debug=True,
                        eps=1e-3,
                        batch_size=params[dataset][test]['batch_size'])
        else:
            dataset = 'monk'
            net = NC([input_units, params[dataset][test]['h1'], output_units], 0, params[dataset][test]['activation'],
                        test,
                        0,
                        params[dataset][test]['epochs'],
                        params[dataset][test]['eta'],
                        params[dataset][test]['activation'],
                        lmbda=params[dataset][test]['lmbda'],
                        momentum=params[dataset][test]['momentum'],
                        debug=True,
                        eps=1e-3,
                        batch_size=params[dataset][test]['batch_size'])

        net.train((X_train, y_train), test_data=(X_test, y_test))
        print(net.best_score())
        # net.plot_grad('gradient')

    