import sys

from src.NN.Network import NR, NC
import src.NN.utils as utils

from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit, train_test_split

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
                    'epochs': [500, 1000],
                    'batch_size': [10, 32, None],
                    'eta':[0.001, 0.01, 0.1],
                    'eps': [1e-4],
                    'optimizer': ['SGM']
                },
            
                'SGD': {
                    'activation': ['Lrelu'],
                    'sizes': [[2], [3], [5]],
                    'lmbda': [0, 0.0001, 0.001, 0.01],
                    'momentum': [0, 0.5, 0.9],
                    'epochs': [500],
                    'batch_size': [10],
                    'eta':[0.01, 0.1],
                    'eps': [1e-4],
                    'optimizer': ['SGD']
                }
            }
        }

        full_name = dataset
        if dataset == 'cup':
            net = NR
            cv = 5
            scoring = 'neg_mean_squared_error'
        else:
            # Removes the monk number
            dataset = 'monk'
            net = NC
            cv = StratifiedShuffleSplit(n_splits=5, test_size=0.20, random_state=42)
            scoring = 'accuracy'
        
        grid = grids[dataset][test]

        gs = GridSearchCV(net(), cv=cv, param_grid=grid, n_jobs=-1, verbose=10, scoring=scoring)
        gs.fit(X_train, y_train)
        print(f"Best score over VL: {gs.best_score_}, best params: {gs.best_params_}\n")

        results = utils.crossValToDf(gs.cv_results_, scoring=scoring)
        results.to_csv(f'./src/NN/res/scores/{full_name}_{test}.csv')  

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
                    'batch_size': 32,
                    'epochs': 1000,
                    'eps': 1e-4,
                    'eta': 0.001,
                    'lmbda': 0.001,
                    'momentum': 0.,
                    'optimizer': test,
                    'sizes': [30, 50],
                    'activation': 'Lrelu',
                    'debug': True,
                },
                'SGM': {
                    'batch_size': 32,
                    'epochs': 1000,
                    'eps':1e-3,
                    'eta': 0.1,
                    'lmbda': 0.001,
                    'optimizer': test,
                    'sizes': [30, 50],
                    'activation': 'Lrelu',
                    'debug': True,
                }
            },
            'monk1': {
                'SGD': {
                    'activation': 'Lrelu',
                    'batch_size': 10,
                    'epochs': 500,
                    'eps': 1e-4,
                    'eta': 0.1,
                    'lmbda': 0.0001,
                    'momentum': 0.9,
                    'optimizer': test,
                    'sizes': [5],
                    'debug': True
                },
                'SGM': {
                    'batch_size': 10,
                    'epochs': 1000,
                    'eps':1e-3,
                    'eta': 0.1,
                    'lmbda': 0.001,
                    'optimizer': test,
                    'sizes': [5],
                    'debug': True,
                }
            },
            'monk2': {
                'SGD': {
                    'activation': 'Lrelu',
                    'batch_size': 10,
                    'epochs': 500,
                    'eps': 1e-4,
                    'eta': 0.1,
                    'lmbda': 0.0001,
                    'momentum': 0.9,
                    'optimizer': test,
                    'sizes': [3],
                    'debug': True
                },
                'SGM': {
                    'batch_size': 10,
                    'epochs': 500,
                    'eps':1e-3,
                    'eta': 0.1,
                    'lmbda': 0.,
                    'optimizer': test,
                    'sizes': [3],
                    'debug': True,
                }
            },
            'monk3': {
                'SGD': {
                    'activation': 'Lrelu',
                    'batch_size': 10,
                    'epochs': 500,
                    'eps': 1e-4,
                    'eta': 0.01,
                    'lmbda': 0.0001,
                    'momentum': 0.5,
                    'optimizer': test,
                    'sizes': [5],
                    'debug': True
                },
                'SGM': {
                    'batch_size': 32,
                    'epochs': 500,
                    'eps':1e-3,
                    'eta': 0.1,
                    'lmbda': 0.01,
                    'optimizer': test,
                    'sizes': [2],
                    'debug': True,
                }
            }
        }

        if dataset == 'cup':
            net = NR(**params[dataset][test])
        else:
            net = NC(**params[dataset][test])

        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        net.fit(X_train, y_train, test_data=(X_test, y_test))
        net.plot_results(f"asdddddd", score=False)
        # net.plot_results(f"{dataset}_{test}", score=False)
        net.plot_grad(f"{dataset}_{test}")
        print(net.best_score())
    