import numpy as np

from src.NN.Network import Network
from src.NN.ActivationFunctions import Sigmoid, Linear
from src.NN.LossFunctions import MeanSquaredError, AccuracyScore


class NC(Network):

    def __init__(self, sizes, seed, activation='sigmoid', lmbda=0.0, momentum=0.5, debug=True):
        super().__init__(sizes, seed, activation, lmbda, momentum, debug)

        # Defines the behavior of the last layer of the network
        self.last_act = Sigmoid
        self.loss = AccuracyScore


    def best_score(self):

        best_score = ()
        if len(self.val_scores) > 0:
            best_score = (np.max(self.val_scores), np.max(self.train_scores))

        return best_score


    def predict(self, data):
        return self.feedforward_batch(data)[2] >= 0.5


class NR(Network):

    def __init__(self, sizes, seed, activation='sigmoid', lmbda=0.0, momentum=0.5, debug=True):
        super().__init__(sizes, seed, activation, lmbda, momentum, debug)

        # Defines the behavior of the last layer of the network
        self.last_act = Linear
        self.loss = MeanSquaredError


    def best_score(self):
        best_score = ()
        if len(self.val_scores) > 0:
            best_score = (np.min(self.val_scores), np.min(self.train_scores))

        return best_score


    def predict(self, data):
        return self.feedforward_batch(data)[2]