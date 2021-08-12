import sys

from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit

from src.NN.Network import NR, NC
import src.utils as utils


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



    # Performs gridsearch over the specified hyperparameters
    if grid:
        
        grids = {
            'cup': {
                'SGM': {    
                    'sizes': [[5, 10], [16, 32], [30, 50]],
                    'lmbda': [0, 0.0001, 0.001, 0.01],
                    'epochs': [1000],
                    'batch_size': [None],
                    'eta':[0.001, 0.005,  0.01, 0.05, 0.1],
                    'eps': [1e-4],
                    'optimizer': ['SGM']
                },
            
                'SGD': {
                    'sizes': [[5, 10], [16, 32], [30, 50]],
                    'lmbda': [0, 0.0001, 0.001, 0.01],
                    'momentum': [0, 0.5, 0.9],
                    'nesterov': [True, False],
                    'epochs': [1000],
                    'batch_size': [None],
                    'eta':[0.001, 0.01, 0.1],
                    'eps': [1e-4],
                    'optimizer': ['SGD']
                }
            },

            'monk': {
                'SGM': {    
                    'sizes': [[5, 10], [16, 32], [30, 50]],
                    'lmbda': [0, 0.0001, 0.001, 0.01],
                    'epochs': [2000],
                    'batch_size': [None],
                    'eta':[0.0001, 0.001, 0.01, 0.1],
                    'eps': [1e-6],
                    'optimizer': ['SGM']
                },
            
                'SGD': {
                    'sizes': [[5, 10], [16, 32], [30, 50]],
                    'lmbda': [0, 0.0001, 0.001, 0.01],
                    'momentum': [0, 0.5, 0.9],
                    'nesterov': [True, False],
                    'epochs': [2000],
                    'batch_size': [None],
                    'eta':[0.0001, 0.001, 0.01, 0.1],
                    'eps': [1e-6],
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
            cv = StratifiedShuffleSplit(n_splits=5, test_size=0.20,
                                        random_state=42)
            scoring = 'accuracy'
        
        grid = grids[dataset][test]

        gs = GridSearchCV(net(), cv=cv, param_grid=grid, n_jobs=-1,
                          verbose=0, scoring=scoring)
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
                    'debug': False,
                },
                'SGM': {
                    'batch_size': None,
                    'epochs': 2000,
                    'eps':1e-6,
                    'eta': 0.05,
                    'lmbda': 0.0001,
                    'optimizer': "SGM",
                    'sizes': [5, 10],
                    'debug': False,
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
                    'debug': False
                },
                'SGM': {
                    'batch_size': None,
                    'epochs': 2000,
                    'eps':1e-6,
                    'eta': 0.1,
                    'lmbda': 0.01,
                    'optimizer': "SGM",
                    'sizes': [16, 32],
                    'debug': False
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
                    'debug': False
                },
                'SGM': {
                    'batch_size': None,
                    'epochs': 2000,
                    'eps':1e-6,
                    'eta': 0.1,
                    'lmbda': 0.,
                    'optimizer': "SGM",
                    'sizes': [16, 32],
                    'debug': False,
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
                    'debug': False
                },
                'SGM': {
                    'batch_size': None,
                    'epochs': 2000,
                    'eps':1e-6,
                    'eta': 0.01,
                    'lmbda': 0.0001,
                    'optimizer': "SGM",
                    'sizes': [5, 10],
                    'debug': False,
                }
            }
        }
        if dataset == 'cup':
            net = NR(**params[dataset][test])
        else:
            net = NC(**params[dataset][test])

        net.fit(X_train, y_train, test_data=(X_test, y_test))
        
        print(net.best_score(name=f"{dataset}_{test}", save=False))
    
