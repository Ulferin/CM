# This class represents the Network used in the CM project and it is
# implemented from scratch following the advices taken during the course
# of ML

import numpy as np
from numpy.random import default_rng
from sklearn.metrics import r2_score
import random

from functions import relu, relu_prime

class Network:
    """This class represents a standard Neural Network, also called Multilayer Perceptron.
    It allows to build a network for both classification and regression tasks.
    """    
    
    def __init__(self, sizes, seed):
        """Initializes the network based on the given :param sizes:.
        Builds the weights and biase vectors for each layer of the network.
        Each layer will be initialized randomly following the normal distribution. 

        :param sizes: Tuple (i, l1, l2, ..., ln, o) containig the number of units
        for each layer, where the first and last elements represents, respectively,
        the input layer and the output layer.
        :param seed: seed for random number generator used for initializing this network
        weights and biases. Needed for reproducibility.        
        """
        rng = default_rng(seed)

        self.num_layers = len(sizes)
        self.sizes = sizes

        # TODO: non possiamo avere un singolo bias per layer invece che un bias per ogni unità?
        #       controllare nel libro dove ha dato questo esempio cosa dice a riguardo
        self.biases = [rng.standard_normal((1,y)) for y in sizes[1:]]
        self.weights = [rng.standard_normal((y,x)) for x, y in zip(sizes[:-1], sizes[1:])]


    def feedforward(self, invec):
        """Applies a feedforward pass to the given input :param in:.
        The output generated by the output unit of this network is returned.

        :param in: network input vector
        :return: network output vector (or scalar)
        """
        out = invec
        for b, w in zip(self.biases[:-1], self.weights[:-1]):
            out = relu(np.dot(w, out.T) + b)

        # Last layer is linear for regression tasks
        return np.dot(self.weights[-1], out.T) + self.biases[-1]


    def SGD(self, training_data, epochs, batch_size, eta, test_data=None):
        """Trains the network using mini-batch stochastic gradient descent,
        applied to the training examples in :param training_data: for a given
        number of epochs and with the specified learning rate. If :param test_data:
        is specified, the learning algorithm will print progresses during the
        training phase.

        :param training_data: training data represented as a numpy ndarray, each row
        represents an example, the last element of each row is the expected output.
        :param epochs: number of epochs for training.
        :param batch_size: number of examples to use at each backward pass.
        :param eta: learning rate.
        :param test_data: optional parameter, used to estimate the performance of the network
        at each phase, defaults to None.
        """        
        if test_data:
            n_test = len(test_data)

        n = len(training_data)
        for e in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+batch_size] for k in range(0, n, batch_size)
            ]

            for mini_batch in mini_batches:
                # Here the code to update the weights and biases for each minibatch
                pass

            if test_data:
                score = self.evaluate(test_data)
                print(f"Epoch {e} completed. Score: {score}")
            else:
                print(f"Epoch {e} completed.")

    
    def evaluate(self, test_data):
        """Evaluates the performances of the Network in the current state,
        propagating the test examples through the network via a complete feedforward
        step. It evaluates the performance using the R2 metric in order to be
        comparable with sklearn out-of-the-box NN results.

        :param test_data: test data to evaluate the NN
        :return: The R2 score as defined by sklearn library
        """        

        preds = [ self.feedforward(x) for x,y in test_data]
        truth = [ y for x,y in test_data ]

        score = r2_score(preds, truth)
        return score
