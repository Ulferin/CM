import sys

import numpy as np
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit

from src.NN.Network import NR, NC

import src.utils as utils
from sklearn.metrics import mean_squared_error, classification_report


def printScoreClas(clf, X, y, score='Accuracy'):
    """
    This function prints the score of classification/regression for all the models but Keras ones. 

    Params:
        - clf: model to use for predictions.
        - X: training samples DataFrame.
        - y: training target variables DataFrame.
        - score : Score measure to print.
    """

    y_pred = clf.predict(X)
    print(f"{score}: {clf.score(X, y)}")
    print(f"MSE: {mean_squared_error(y, y_pred)}")
    if score=='Accuracy':
        print(classification_report(y, y_pred))

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
        'sgd': {
            'activation': 'logistic',
            'alpha': 0.3,
            'batch_size': None,
            'hidden_layer_sizes': [5, 10],
            'learning_rate_init': 0.01,
            'max_iter': 10000,
            'momentum': 0.5,
            'solver': 'sgd',
            'tol': 1e-08
        },
        'adam': {
            'activation': 'logistic',
            'alpha': 0.5,
            'batch_size': None,
            'hidden_layer_sizes': [5, 10],
            'learning_rate_init': 0.005,
            'max_iter': 10000,
            'solver': 'adam',
            'tol': 1e-08
        }
    },
    'monk1': {
        'sgd': {
            'activation': 'logistic',
            'alpha': 0.001,
            'batch_size': 10,
            'hidden_layer_sizes': [3, 5],
            'learning_rate_init': 0.005,
            'max_iter': 10000,
            'momentum': 0.9,
            'nesterovs_momentum': False,
            'solver': 'sgd',
            'tol': 1e-06
        },
        'adam': {
            'activation': 'logistic',
            'alpha': 0.005,
            'batch_size': None,
            'hidden_layer_sizes': [5,10],
            'learning_rate_init': 0.001,
            'max_iter': 10000,
            'solver': "adam",
            'tol': 1e-6,
        }
    },
    'monk2': {
        'sgd': {
            'activation': 'logistic',
            'alpha': 0.001,
            'batch_size': 10,
            'hidden_layer_sizes': [3, 5],
            'learning_rate_init': 0.1,
            'max_iter': 10000,
            'momentum': 0.5,
            'nesterovs_momentum': False,
            'solver': 'sgd',
            'tol': 1e-06
        },
        'adam': {
            'activation': 'logistic',
            'alpha': 0.001,
            'batch_size': 10,
            'hidden_layer_sizes': [3, 5],
            'learning_rate_init': 0.001,
            'max_iter': 10000,
            'solver': 'adam',
            'tol': 1e-6
        }
    },
    'monk3': {
        'sgd': {
            'activation': 'logistic',
            'alpha': 0.001,
            'batch_size': None,
            'hidden_layer_sizes': [2, 3],
            'learning_rate_init': 0.1,
            'max_iter': 10000,
            'momentum': 0.5,
            'nesterovs_momentum': False,
            'solver': 'sgd',
            'tol': 1e-06
        },
        'adam': {
            'activation': 'logistic',
            'alpha': 0.001,
            'batch_size': None,
            'hidden_layer_sizes': [2, 3],
            'learning_rate_init': 0.001,
            'max_iter': 10000,
            'solver': 'adam',
            'tol': 1e-6
        }
    }
}


# Parameter grids for gridsearch test
grids = {
    'cup': {

        'sgd': {
            'hidden_layer_sizes': [[3,5], [5,3], [10,5], [5, 10]],
            'alpha': [0.1, 0.2, 0.3],
            'momentum': [0.5, 0.9],
            'nesterovs_momentum': [True],
            'max_iter': [10000],
            'batch_size': [None],
            'learning_rate_init':[0.01, 0.02, 0.05, 0.1],
            'tol': [1e-4],
            'activation': ['logistic'],
            'solver': ['sgd']
        },

        'adam': {
            'hidden_layer_sizes': [[3,5], [5,3], [10,5], [5, 10]],
            'alpha': [0.1, 0.2, 0.3, 0.5],
            'max_iter': [10000],
            'batch_size': [None],
            'learning_rate_init':[0.005, 0.01, 0.02, 0.05],
            'tol': [1e-4],
            'activation': ['logistic'],
            'solver': ['adam']
        }
    },

    'monk': {

        'sgd': {
            'hidden_layer_sizes': [[2,3], [3,5]],
            'alpha': [0.001, 0.005, 0.1],
            'momentum': [0, 0.5, 0.9],
            'nesterovs_momentum': [False, True],
            'max_iter': [3000],
            'batch_size': [10, 32, None],
            'learning_rate_init':[0.001, 0.005, 0.1],
            'tol': [1e-6],
            'solver': ['sgd'],
            'activation': ['logistic'],
        },

        'adam': {
            'hidden_layer_sizes': [[2,3], [3,5], [5, 10]],
            'alpha': [0.001, 0.005, 0.1],
            'max_iter': [5000],
            'batch_size': [10, 32, None],
            'learning_rate_init':[0.001, 0.005, 0.1],
            'tol': [1e-6],
            'activation': ['logistic'],
            'solver': ['adam']
        }
    }
}


if __name__ == '__main__':
    test = sys.argv[1]      # Test type, either 'CM', 'NAG' or 'adam' when
                            # grid==false, otherwise 'sgd' or 'adam'
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
            net = NR
            # net = MLPRegressor
            cv = 5
            scoring = 'neg_mean_squared_error'
        else:
            # Removes the monk number
            dataset = 'monk'
            net = NC
            # net = MLPClassifier
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
        plot_name = f"{dataset}_{test}"
        solver = test
        # Add nesterovs_momentum momentum to params
        if test == 'NAG':
            test = 'sgd'
            params[dataset][test]['nesterovs_momentum'] = True
        if test == 'CM':
            test = 'sgd'
            params[dataset][test]['nesterovs_momentum'] = False

        if dataset == 'cup':
            net = NR(**params[dataset][test], verbose=True)
            net_sk = MLPRegressor(**params[dataset][test], verbose=True)
            # params[dataset][test]['max_iter'] = 10000
            net_eval = NR(**params[dataset][test], verbose=True)
        else:
            net = NC(**params[dataset][test], verbose=True)
            # net_sk = MLPClassifier(**params[dataset][test], verbose=True)
            # params[dataset][test]['max_iter'] = 1000
            # params[dataset][test]['tol'] = 1e-16
            net_eval = NC(**params[dataset][test], verbose=True)
        print("Evaluating f_* ...")
        net_eval.fit(X_train, y_train, test_data=(X_test, y_test))
        net.fit(X_train, y_train, test_data=(X_test, y_test), f_star_set=net_eval.f_star, grad_star=net_eval.grad_star)
        # net_sk.fit(X_train, y_train.ravel())

        net.plot_gap(dataset, solver, save=True)
        net.plot_rate(plot_name, True)
        net.plot_grad(plot_name, True, False, False)
        net.plot_grad(plot_name, True, False, True)
        net.plot_results(plot_name, False, True)
        print(f"mean rate: {np.mean(net.conv_rate)}")
        print(f"f_*: {net_eval.f_star}")
        print(f"last gap: {net.gap[-1]}")
        print("Improved network:")
        # printScoreClas(net_sk, X_train, y_train.ravel())
        print(net.best_score(name=plot_name, save=False))
