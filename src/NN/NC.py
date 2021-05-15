import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error

from Network import Network
from ActivationFunctions import Sigmoid, Linear
from LossFunctions import MeanSquaredError


class NC(Network):

    def __init__(self, sizes, seed, activation='sigmoid', lmbda=0.0, momentum=0.5, debug=True):
        super().__init__(sizes, seed, activation, lmbda, momentum, debug)

        # Defines the behavior of the last layer of the network
        self.last_act = Sigmoid


    def best_score(self):
        return (np.max(self.val_scores), np.max(self.train_scores))


    def evaluate(self, test_data, train_data):
        """Evaluates the performances of the Network in the current state,
        propagating the test examples through the network via a complete feedforward
        step. It evaluates the performance using the R2 metric in order to be
        comparable with sklearn out-of-the-box NN results.

        :param test_data: test data to evaluate the NN
        :return: The overall accuracy for the current prediction
        """        

        score = []

        preds_test = [ np.array(self.feedforward(x)[2] > 0.5).reshape(y.shape) for x,y in test_data]
        truth_test = [ y for x,y in test_data ]

        preds_train = [np.array(self.feedforward(x)[2] > 0.5).reshape(y.shape) for x,y in train_data]
        truth_train = [y for x,y in train_data]

        score.append(accuracy_score(truth_test, preds_test))
        score.append(accuracy_score(truth_train, preds_train))

        return score


class NR(Network):

    def __init__(self, sizes, seed, activation='sigmoid', lmbda=0.0, momentum=0.5, debug=True):
        super().__init__(sizes, seed, activation, lmbda, momentum, debug)

        # Defines the behavior of the last layer of the network
        self.last_act = Linear


    def best_score(self):
        return (np.min(self.val_scores), np.min(self.train_scores))

    def evaluate(self, test_data:tuple, train_data:tuple):
            # TODO: cambiare descrizione di evaluate nelle due sottoclassi
        """Evaluates the performances of the Network in the current state,
        propagating the test examples through the network via a complete feedforward
        step. It evaluates the performance using the R2 metric in order to be
        comparable with sklearn out-of-the-box NN results.

        :param test_data: test data to evaluate the NN
        :return: The mean squared error for the current prediction
        """        
        score_test = []
        score_train = []

        preds_test = [np.array(self.feedforward(x)[2]).reshape(y.shape) for x,y in zip(test_data[0], test_data[1])]
        truth_test = [y for y in test_data[1]]

        preds_train = [np.array(self.feedforward(x)[2]).reshape(y.shape) for x,y in zip(train_data[0], train_data[1])]
        truth_train = [y for y in train_data[1]]
        
        score_test.append(MeanSquaredError.loss(truth_test, preds_test))
        score_train.append(MeanSquaredError.loss(truth_train, preds_train))

        return (score_test, score_train), preds_train