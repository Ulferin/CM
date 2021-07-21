import sys

from src.NN.Network import NR, NC
import src.NN.utils as utils

from sklearn.model_selection import GridSearchCV, StratifiedKFold

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
        
        # TODO: mettere in file di conf
        grids = {
            'cup': {
                'SGM': {    
                    'sizes': [[16, 32], [30, 50], [50, 50]],
                    'lmbda': [0, 0.001, 0.01],
                    'activation': ['Lrelu'],
                    'debug': [False],
                    'epochs': [500, 1000],
                    'batch_size': [32, None],
                    'eta':[0.001, 0.01, 0.1],
                    'eps': [1e-4],
                    'optimizer': ['SGM']
                },
            
                'SGD': {
                    'sizes': [[16, 32], [30, 50], [50, 50]],
                    'lmbda': [0, 0.001, 0.01],
                    'momentum': [0, 0.2, 0.5, 0.9],
                    'activation': ['Lrelu'],
                    'debug': [False],
                    'epochs': [500, 1000],
                    'batch_size': [32, None],
                    'eta':[0.0001, 0.001],
                    'eps': [1e-4],
                    'optimizer': ['SGD']
                }
            },

            'monk': {
                'SGM': {    
                    'sizes': [[2], [3], [5]],
                    'lmbda': [0, 0.001, 0.01],
                    'activation': ['Lrelu'],
                    'debug': [False],
                    'epochs': [500],
                    'batch_size': [32, None],
                    'eta':[0.001, 0.01, 0.1],
                    'eps': [1e-4],
                    'optimizer': ['SGM']
                },
            
                'SGD': {
                    'sizes': [[16, 32], [30, 50], [50, 50]],
                    'lmbda': [0, 0.001, 0.01],
                    'momentum': [0, 0.2, 0.5, 0.9],
                    'activation': ['Lrelu'],
                    'debug': [False],
                    'epochs': [500, 1000],
                    'batch_size': [32, None],
                    'eta':[0.0001, 0.001],
                    'eps': [1e-4],
                    'optimizer': ['SGD']
                }
            }
        }

        if dataset == 'cup':
            net = NR
            cv = 5
            scoring = 'neg_mean_squared_error'
        else:
            # Removes the monk number
            dataset = 'monk'
            net = NC
            cv = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
            scoring = 'accuracy'
        
        grid = grids[dataset][test]

        gs = GridSearchCV(net(), cv=5, param_grid=grid, n_jobs=-1, verbose=10, scoring=scoring)
        gs.fit(X_train, y_train)
        print(f"Best score over VL: {gs.best_score_}, best params: {gs.best_params_}\n")

        # Retraining w/ best parameters
        print("Retraining network with best parameters:")
        net = net(**gs.best_params_)
        net.fit(X_train, y_train, test_data=(X_test, y_test))
        print(net.best_score())
    
    else:

        # TODO: aggiungere json di configurazione
        params = {
            'cup': {
                'SGD': {
                    'activation': 'Lrelu',
                    'sizes': [30, 50],
                    'lmbda': 0.001,
                    'momentum': 0.,
                    'epochs': 1000,
                    'batch_size': 32,
                    'eta': 0.001,
                    'optimizer': test,
                    'debug': True,
                    'eps': 1e-4,
                },
                'SGM': {
                    'activation': 'Lrelu',
                    'sizes': [30, 50],
                    'lmbda': 0.001,
                    'momentum': 0.,
                    'epochs': 1000,
                    'batch_size': 32,
                    'eta': 0.1,
                    'optimizer': test,
                    'debug': True,
                    'eps':1e-3
                }
            },
            'monk': {
                'SGD': {
                    'activation': 'Lrelu',
                    'batch_size': 10,
                    'debug': True,
                    'epochs': 1000,
                    'eps': 1e-4,
                    'eta': 0.1,
                    'lmbda': 0.05,
                    'momentum': 0.9,
                    'optimizer': test,
                    'sizes': [5],
                    'debug': True
                },
                'SGM': {
                    'sizes': [3],
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


        net.fit(X_train, y_train, test_data=(X_test, y_test))
        print(net.best_score())
        # net.plot_grad('gradient')

    