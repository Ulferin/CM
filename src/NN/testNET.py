import numpy as np
import sys

from src.NN.NC import NR, NC
import src.NN.utils as utils

datasets = {
    'cup': 'data/ML-CUP20-TR.csv',
    'monk1': 'data/monks-1',
    'monk2': 'data/monks-2', 
    'monk3': 'data/monks-3', 
}


if __name__ == '__main__':
    test = sys.argv[1]
    dataset = sys.argv[2]

    if dataset == 'cup':
        X_train, X_test, y_train, y_test = utils.load_CUP(datasets[dataset])
    else:
        X_train, X_test, y_train, y_test = utils.load_monk(datasets[dataset])
        X_train, X_test = utils.prepare_data(X_train, X_test)

    # Loads the input and output layers shape
    input_units = X_train.shape[1]
    output_units = y_train.shape[1] if len(y_train.shape) == 2 else 1

    epochs = [200, 400, 600]
    hidden1 = [2, 5, 10, 20]
    hidden2 = [2, 5, 10, 20]
    batch = [10, 30]
    eta = [0.00001, 0.0001, 0.001, 0.01, 0.1]
    lmbda = [0.00001, 0.001]
    momentum = [0.5, 0.9]

    # Performs gridsearch over the specified hyperparameters
    if test == 'grid':
        for ep in epochs:
            for h1 in hidden1:
                for h2 in hidden2:
                    for b in batch:
                        for e in eta:
                            for l in lmbda:
                                for m in momentum:
                                    
                                    if dataset == 'cup':
                                        net = NR([input_units, h1, h2, output_units], 0, 'Lrelu', lmbda=l, momentum=m, debug=False)
                                    else:
                                        net = NC([input_units, h1, output_units], 0, 'Lrelu', lmbda=l, momentum=m, debug=False)

                                    net.SGD((X_train.reshape(X_train.shape[0],1,-1), y_train), epochs=ep, batch_size=b, eta=e, test_data=(X_test.reshape(X_test.shape[0],1,-1), y_test))
                                    print(f"The best score for ep:{ep}, h1:{h1}, h2:{h2}, b:{b}, e:{e}, l:{l}, m:{m} was: {net.best_score()}")
                                    net.plot_score(f"test_np/{dataset}")

    else:

        params = {
            'cup': {
                'SGD': {
                    'h1': 16,
                    'h2': 32,
                    'activation': 'Lrelu',
                    'lmbda': 0.1,
                    'momentum': 0.7,
                    'epochs': 500,
                    'batch_size': 64,
                    'eta': 0.001,
                },
                'SGM': {
                    'h1': 16,
                    'h2': 32,
                    'activation': 'Lrelu',
                    'lmbda': 0.1,
                    'momentum': 0.7,
                    'epochs': 500,
                    'batch_size': 64,
                    'eta': 11
                }
            },
            'monk': {
                'SGD': {
                    'h1': 3,
                    'h2': None,
                    'activation': 'Lrelu',
                    'lmbda': 0.2,
                    'momentum': 0.9,
                    'epochs': 500,
                    'batch_size': 32,
                    'eta': 0.1
                },
                'SGM': {
                    'h1': 3,
                    'h2': None,
                    'activation': 'Lrelu',
                    'lmbda': 0.2,
                    'momentum': 0.9,
                    'epochs': 500,
                    'batch_size': 32,
                    'eta': 0.1
                }
            }
        }

        if dataset == 'cup':
            net = NR([input_units, params[dataset][test]['h1'], params[dataset][test]['h2'], output_units], 0, params[dataset][test]['activation'], lmbda=params[dataset][test]['lmbda'], momentum=params[dataset][test]['momentum'], debug=False)
        else:
            dataset = 'monk'
            net = NC([input_units, params[dataset][test]['h1'], output_units], 0, params[dataset][test]['activation'], lmbda=params[dataset][test]['lmbda'], momentum=params[dataset][test]['momentum'], debug=False)

        net.train(test, (X_train, y_train), epochs=params[dataset][test]['epochs'], batch_size=params[dataset][test]['batch_size'], eta=params[dataset][test]['eta'], test_data=(X_test, y_test))
        print(f"The best score for ep:{params[dataset][test]['epochs']}, h1:{params[dataset][test]['h1']}, h2:{params[dataset][test]['h2']}, b:{params[dataset][test]['batch_size']}, e:{params[dataset][test]['eta']}, l:{params[dataset][test]['lmbda']}, m:{params[dataset][test]['momentum']} was: {net.best_score()}")
        # net.plot_grad('gradient')

    