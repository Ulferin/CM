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
        
        # TODO: mettere in file di conf
        grids = {
            'cup': {
                'SGM': {    
                    'sizes': [[16, 32], [30, 50], [50, 50]],
                    'lmbda': [0, 0.0001, 0.001, 0.01],
                    'epochs': [1000],
                    'batch_size': [32, None],
                    'eta':[0.001, 0.01],
                    'eps': [1e-4],
                    'optimizer': ['SGM']
                },
            
                'SGD': {
                    'sizes': [[16, 32], [30, 50], [50, 50]],
                    'lmbda': [0, 0.0001, 0.001, 0.01],
                    'momentum': [0, 0.2, 0.5, 0.9],
                    'nesterov': [True, False],
                    'epochs': [1000],
                    'batch_size': [32, None],
                    'eta':[0.001, 0.01],
                    'eps': [1e-4],
                    'optimizer': ['SGD']
                }
            },

            'monk': {
                'SGM': {    
                    'sizes': [[2], [3], [5]],
                    'lmbda': [0, 0.0001, 0.001, 0.01],
                    'epochs': [2000],
                    'batch_size': [32, None],
                    'eta':[0.01, 0.1, 0.3, 0.5, 0.6, 0.7],
                    'eps': [1e-6],
                    'optimizer': ['SGM']
                },
            
                'SGD': {
                    'activation': ['Lrelu'],
                    'sizes': [[2], [3], [5]],
                    'lmbda': [0, 0.0001, 0.001, 0.01],
                    'momentum': [0, 0.5, 0.9],
                    'nesterov': [True, False],
                    'epochs': [2000],
                    'batch_size': [32, None],
                    'eta':[0.01, 0.1, 0.5, 0.7],
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
            # Removing regularization for MONK1 and MONK2
            # if dataset != 'monk3': grids['monk'][test]['lmbda'] = [0.]
            
            # Removes the monk number
            dataset = 'monk'
            net = NC
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

        params = {
            'cup': {
                'SGD': {
                    'batch_size': 32,
                    'epochs': 1000,
                    'eps': 1e-6,
                    'eta': 0.001,
                    'lmbda': 0.001,
                    'momentum': 0.,
                    'optimizer': "SGD",
                    'sizes': [30, 50],
                    'activation': 'Lrelu',
                    'debug': True,
                },
                'SGM': {
                    'batch_size': 32,
                    'epochs': 1000,
                    'eps':1e-6,
                    'eta': 0.1,
                    'lmbda': 0.001,
                    'optimizer': "SGM",
                    'sizes': [30, 50],
                    'activation': 'Lrelu',
                    'debug': True,
                }
            },
            'monk1': {
                'SGD': {
                    'activation': 'Lrelu',
                    'batch_size': 32,
                    'epochs': 1000,
                    'eps': 1e-6,
                    'eta': 0.1,
                    'lmbda': 0.01,
                    'momentum': 0.9,
                    'optimizer': "SGD",
                    'sizes': [5],
                    'nesterov': True,
                    'debug': True
                },
                'SGM': {
                    'batch_size': None,
                    'epochs': 1000,
                    'eps':1e-6,
                    'eta': 0.1,
                    'lmbda': 0.01,
                    'optimizer': "SGM",
                    'sizes': [5],
                    'debug': True
                }
            },
            'monk2': {
                'SGD': {
                    'activation': 'Lrelu',
                    'batch_size': 32,
                    'epochs': 1000,
                    'eps': 1e-6,
                    'eta': 0.1,
                    'lmbda': 0.01,
                    'momentum': 0.5,
                    'optimizer': "SGD",
                    'sizes': [3],
                    'nesterov': True,
                    'debug': True
                },
                'SGM': {
                    'batch_size': 10,
                    'epochs': 1000,
                    'eps':1e-6,
                    'eta': 0.1,
                    'lmbda': 0.001,
                    'optimizer': "SGM",
                    'sizes': [3],
                    'debug': True,
                }
            },
            'monk3': {
                'SGD': {
                    'activation': 'Lrelu',
                    'batch_size': 10,
                    'epochs': 1000,
                    'eps': 1e-6,
                    'eta': 0.1,
                    'lmbda': 0.01,
                    'momentum': 0.9,
                    'nesterov': True,
                    'optimizer': "SGD",
                    'sizes': [5],
                    'debug': True
                },
                'SGM': {
                    'batch_size': None,
                    'epochs': 1000,
                    'eps':1e-6,
                    'eta': 0.1,
                    'lmbda': 0.01,
                    'optimizer': "SGM",
                    'sizes': [5],
                    'debug': True,
                }
            }
        }
        if dataset == 'cup':
            net = NR(**params[dataset][test])
        else:
            net = NC(**params[dataset][test])

        net.fit(X_train, y_train, test_data=(X_test, y_test))

#         net.plot_results(f"{dataset}_{test}", score=False, time=True)
        # net.plot_results(f"{dataset}_{test}", score=False, time=False)
        # net.plot_results(f"{dataset}_{test}", score=True, time=True)
        # net.plot_results(f"{dataset}_{test}", score=True, time=False)
        # net.plot_grad(f"{dataset}_{test}")
        
        print(net.best_score(name=f"{dataset}_{test}", save=True))
    
