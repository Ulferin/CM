import sys

from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit

from src.NN.net_old import NR as NR_old, NC as NC_old
from src.NN.Network import NR, NC

import src.utils as utils


datasets = {
    'cup': 'data/ML-CUP20-TR.csv',
    'monk1': 'data/monks-1',
    'monk2': 'data/monks-2', 
    'monk3': 'data/monks-3', 
}

# Hyperparameters configurations for each model/dataset couple
# the actual configuration comes from the results of the performed gridsearches
# as described in the report file.
params = {
    'cup': {
        'SGD': {
            'batch_size': None,
            'epochs': 2000,
            'eps': 1e-6,
            'eta': 0.001,
            'lmbda': 0.0001,
            'momentum': 0.9,
            'optimizer': "SGD",
            'sizes': [30, 50],
        },
        'Adam': {
            'batch_size': None,
            'epochs': 2000,
            'eps': 1e-6,
            'eta': 0.1,
            'lmbda': 0.01,
            'momentum': 0.5,
            'optimizer': "Adam",
            'sizes': [16, 32],
        }
    },
    'monk1': {
        'SGD': {
            'batch_size': None,
            'epochs': 2000,
            'eps': 1e-6,
            'eta': 0.1,
            'lmbda': 0.01,
            'momentum': 0.5,
            'optimizer': "SGD",
            'sizes': [16, 32],
        },
        'Adam': {
            'batch_size': None,
            'epochs': 2000,
            'eps': 1e-6,
            'eta': 0.1,
            'lmbda': 0.0001,
            'optimizer': "Adam",
            'sizes': [16, 32],
        }
    },
    'monk2': {
        'SGD': {
            'batch_size': None,
            'epochs': 2000,
            'eps': 1e-6,
            'eta': 0.1,
            'lmbda': 0.,
            'momentum': 0.5,
            'optimizer': "SGD",
            'sizes': [16, 32],
        },
        'Adam': {
            'batch_size': None,
            'epochs': 2000,
            'eps': 1e-6,
            'eta': 0.1,
            'lmbda': 0.01,
            'optimizer': "Adam",
            'sizes': [16, 32],
        }
    },
    'monk3': {
        'SGD': {
            'batch_size': None,
            'epochs': 2000,
            'eps': 1e-6,
            'eta': 0.01,
            'lmbda': 0.0001,
            'momentum': 0.5,
            'optimizer': "SGD",
            'sizes': [5, 10],
        },
        'Adam': {
            'batch_size': None,
            'epochs': 2000,
            'eps': 1e-6,
            'eta': 0.1,
            'lmbda': 0.01,
            'optimizer': "Adam",
            'sizes': [16, 32],
        }
    }
}


# Parameter grids for gridsearch test
grids = {
    'cup': {
        
        'SGD': {
            'sizes': [[2,5], [16, 32], 0],
            'lmbda': [0, 0.0001, 0.001, 0.01],
            'momentum': [0.5, 0.9],
            'nesterov': [True, False],
            'epochs': [2000],
            'batch_size': [None],
            'eta':[0.001, 0.01, 0.1],
            'eps': [1e-4],
            'optimizer': ['SGD']
        },

        'Adam': {
            'sizes': [[16, 32]],
            'lmbda': [0.0001, 0.001, 0.01],
            'epochs': [1000],
            'batch_size': [None],
            'eta':[0.001, 0.01, 0.1],
            'eps': [1e-4],
            'optimizer': ['Adam']
        }
    },

    'monk': {
    
        'SGD': {
            'sizes': [[5, 10], [16, 32], [30, 50]],
            'lmbda': [0, 0.0001, 0.001],
            'momentum': [0, 0.5, 0.9],
            'nesterov': [True, False],
            'epochs': [2000],
            'batch_size': [None],
            'eta':[0.0001, 0.001, 0.01, 0.1],
            'eps': [1e-6],
            'optimizer': ['SGD'],
            'activation': ['Lrelu']
        },

        'Adam': {
            'sizes': [[1,2], [3,5], [5, 10]],
            'lmbda': [0, 0.0001, 0.001, 0.01],
            'epochs': [500],
            'batch_size': [10, 32],
            'eta':[0.0001, 0.001, 0.01, 0.1],
            'eps': [1e-4],
            'optimizer': ['Adam']
        }
    }
}


if __name__ == '__main__':
    test = sys.argv[1]      # Test type, either 'CM', 'NAG' or 'SGM' when
                            # grid==false, otherwise 'SGD' or 'SGM'
    dataset = sys.argv[2]   # Dataset to use, 'monk#' or 'grid'
    grid = (len(sys.argv) > 3
            and sys.argv[3] == 'grid') # Whether to perform grid or not

    # Load specified dataset
    if dataset == 'cup':
        X_train, X_test, y_train, y_test = utils.load_CUP(datasets[dataset])
    else:
        X_train, X_test, y_train, y_test = utils.load_monk(datasets[dataset])


    # Performs gridsearch over the specified hyperparameters
    if grid:

        full_name = dataset
        if dataset == 'cup':
            net = NR_old if sys.argv[4] == 'old' else NR
            cv = 5
            scoring = 'neg_mean_squared_error'
        else:
            # Removes the monk number
            dataset = 'monk'
            net = NC_old if sys.argv[4] == 'old' else NC
            cv = StratifiedShuffleSplit(n_splits=5, test_size=0.20,
                                        random_state=42)
            scoring = 'accuracy'
        
        grid = grids[dataset][test]

        gs = GridSearchCV(net(), cv=cv, param_grid=grid, n_jobs=-1,
                          verbose=10, scoring=scoring)
        gs.fit(X_train, y_train)
        print(f"Best score over VL: {gs.best_score_}, "
              f"best params: {gs.best_params_}\n")

        results = utils.crossValToDf(gs.cv_results_, scoring=scoring)
        results.to_csv(f'./src/NN/res/scores/{full_name}_{test}.csv')  

        # Retraining w/ best parameters
        print("Retraining network with best parameters:")
        net = net(**gs.best_params_)
        net.fit(X_train, y_train, test_data=(X_test, y_test))
        print(net.best_score())
    
    else:
        # Add nesterov momentum to params
        if test == 'NAG':
            test = 'SGD'
            params[dataset][test]['nesterov'] = True
        if test == 'CM':
            test = 'SGD'

        if dataset == 'cup':
            net = NR(**params[dataset][test], debug=True)
            net_old = NR_old(**params[dataset][test], debug=True)
        else:
            net = NC(**params[dataset][test], debug=True)
            net_old = NC_old(**params[dataset][test], debug=True)

        net.fit(X_train, y_train, test_data=(X_test, y_test))
        net_old.fit(X_train, y_train, test_data=(X_test, y_test))
        
        print("Improved network:")
        print(net.best_score(name=f"{dataset}_{test}", save=False))
        print("Old network:")
        print(net_old.best_score(name=f"{dataset}_{test}", save=False))
    
