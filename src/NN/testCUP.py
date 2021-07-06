import numpy as np
import sys


from src.NN.NC import NR
import src.NN.utils as utils


X_train, X_test, y_train, y_test = utils.load_CUP("data/ML-CUP20-TR.csv")


# Loads the input and output layers shape
input_units = X_train.shape[1]
output_units = y_train.shape[1]

# Builds the training data for the NN
training_data = X_train.reshape(X_train.shape[0], 1, -1)
test_data = X_test.reshape(X_test.shape[0], 1, -1)

epochs = [200, 400, 600]
hidden1 = [2, 5, 10, 20]
hidden2 = [2, 5, 10, 20]
batch = [10, 30]
eta = [0.00001, 0.0001, 0.001, 0.01, 0.1]
lmbda = [0.00001, 0.001]
momentum = [0.5, 0.9]


if __name__ == '__main__':
    test = sys.argv[1]

    h1 = 16
    h2 = 32
    activation = 'Lrelu'
    lmbda = 0
    momentum = 0.5
    epochs = 3000
    batch_size = 32
    eta = 0.001

    if test == 'grid':
        for ep in epochs:
            for h1 in hidden1:
                for h2 in hidden2:
                    for b in batch:
                        for e in eta:
                            for l in lmbda:
                                for m in momentum:
                                    net = NR([input_units, h1, h2, output_units], 0, 'relu', lmbda=l, momentum=m, debug=False)
                                    net.SGD((training_data, y_train), epochs=ep, batch_size=b, eta=e, test_data=(test_data, y_test))
                                    print(f"The best score for ep:{ep}, h1:{h1}, h2:{h2}, b:{b}, e:{e}, l:{l}, m:{m} was: {net.best_score()}")
                                    net.plot_score(f"test_np/cup")

    elif test == 'std':
        net = NR([input_units, h1, h2, output_units], 0, activation, lmbda=lmbda, momentum=momentum, debug=False)
        net.SGD((training_data, y_train), epochs=epochs, batch_size=batch_size, eta=eta, test_data=(test_data, y_test))
        print(f"The best score for ep:{epochs}, h1:{h1}, h2:{h2}, b:{batch_size}, e:{eta}, l:{lmbda}, m:{momentum} was: {net.best_score()}")
        net.plot_score(f"test_np/cup")

    elif test == 'sub':
        net = NR([input_units, 2, 3, output_units], 0, activation, lmbda=lmbda, momentum=momentum, debug=False)
        net.subgrad((training_data, y_train), epochs=5000, start=11)

    