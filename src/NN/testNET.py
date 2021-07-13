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
        h1 = 79
        h2 = 61
        activation = 'Lrelu'
        lmbda = 0.2
        momentum = 0.9
        epochs = 500
        batch_size = 128
        eta = 0.001

        if test == 'std':

            if dataset == 'cup':
                net = NR([input_units, h1, h2, output_units], 0, activation, lmbda=lmbda, momentum=momentum, debug=False)
            else:
                net = NC([input_units, h1, output_units], 0, activation, lmbda=lmbda, momentum=momentum, debug=False)

            net.SGD((X_train, y_train), epochs=epochs, batch_size=batch_size, eta=eta, test_data=(X_test, y_test))
            print(f"The best score for ep:{epochs}, h1:{h1}, h2:{h2}, b:{batch_size}, e:{eta}, l:{lmbda}, m:{momentum} was: {net.best_score()}")
            net.plot_grad('gradient')

        elif test == 'sub':
            net = NR([input_units, h1, h2, output_units], 0, activation, lmbda=lmbda, momentum=momentum, debug=False)
            net.subgrad((X_train, y_train), epochs=epochs, batch_size=batch_size, start=5, test_data=(X_test, y_test))

    