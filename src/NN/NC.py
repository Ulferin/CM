import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error

from Network import Network
from functions import sigmoid, sigmoid_prime


class NC(Network):

    def __init__(self, sizes, seed, activation='sigmoid', lmbda=0.0, momentum=0.5, debug=True):
        super().__init__(sizes, seed, activation, lmbda, momentum, debug)

        # Defines the behavior of the last layer of the network
        self.last_act = sigmoid
        self.last_der = sigmoid_prime


    def best_score(self):
        return np.max(self.scores)


    def evaluate(self, test_data):
        """Evaluates the performances of the Network in the current state,
        propagating the test examples through the network via a complete feedforward
        step. It evaluates the performance using the R2 metric in order to be
        comparable with sklearn out-of-the-box NN results.

        :param test_data: test data to evaluate the NN
        :return: The overall accuracy for the current prediction
        """        

        preds = [ np.array(self.feedforward(x)[2] > 0.5).reshape(y.shape) for x,y in test_data]
        truth = [ y for x,y in test_data ]

        score = accuracy_score(truth, preds)
        return score


class NR(Network):

    def __init__(self, sizes, seed, activation='sigmoid', lmbda=0.0, momentum=0.5, debug=True):
        super().__init__(sizes, seed, activation, lmbda, momentum, debug)

        # Defines the behavior of the last layer of the network
        self.last_act = lambda x: x
        self.last_der = lambda x: 1


    def best_score(self):
        return np.min(self.scores)


    def evaluate(self, test_data):
        """Evaluates the performances of the Network in the current state,
        propagating the test examples through the network via a complete feedforward
        step. It evaluates the performance using the R2 metric in order to be
        comparable with sklearn out-of-the-box NN results.

        :param test_data: test data to evaluate the NN
        :return: The mean squared error for the current prediction
        """        

        preds = [ np.array(self.feedforward(x)[2]).reshape(y.shape) for x,y in test_data]
        truth = [ y for x,y in test_data ]

        # print(f"exp: {truth[1]}, pred: {preds[1]}")
        score = mean_squared_error(truth, preds)
        return score