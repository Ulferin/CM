import numpy as np
import sys


from NC import NR
import utils


X_train, X_test, y_train, y_test = utils.load_CUP("../../data/ML-CUP20-TR.csv")


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

    for ep in epochs:
        for h1 in hidden1:
            for h2 in hidden2:
                for b in batch:
                    for e in eta:
                        for l in lmbda:
                            for m in momentum:
                                net = NR([input_units, h1, h2, output_units], 0, 'relu2', lmbda=l, momentum=m, debug=False)
                                net.SGD((training_data, y_train), epochs=ep, batch_size=b, eta=e, test_data=(test_data, y_test))
                                print(f"The best score for ep:{ep}, h1:{h1}, h2:{h2}, b:{b}, e:{e}, l:{l}, m:{m} was: {net.best_score()}")
                                net.plot_score(f"test_np/cup")

    