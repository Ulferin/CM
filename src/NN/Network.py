# This class represents the Network used in the CM project and it is
# implemented from scratch following the advices taken during the course
# of ML

from abc import ABCMeta, abstractmethod
import time
import random

import numpy as np
from numpy import linalg
from numpy.linalg import norm
from numpy.random import default_rng

from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from sklearn.utils import shuffle

from matplotlib import pyplot as plt

from src.NN.ActivationFunctions import ReLU, Sigmoid, LeakyReLU
from src.NN.optimizers import SGD, SGM


ACTIVATIONS = {
    'relu': ReLU,
    'Lrelu':LeakyReLU,
    'sigmoid': Sigmoid
}

OPTIMIZERS = {
    'SGD': SGD,
    'SGM': SGM
}


class Network(metaclass=ABCMeta):
    """This class represents a standard Neural Network, also called Multilayer Perceptron.
    It allows to build a network for both classification and regression tasks.
    """    
    
    @abstractmethod
    def __init__(self, sizes, seed, activation='sigmoid', lmbda=0.0, momentum=0.0, debug=True):
        """Initializes the network based on the given :param sizes:.
        Builds the weights and biase vectors for each layer of the network.
        Each layer will be initialized randomly following the normal distribution. 

        :param sizes: Tuple (i, l1, l2, ..., ln, o) containig the number of units
            for each layer, where the first and last elements represents, respectively,
            the input layer and the output layer.
        :param seed: seed for random number generator used for initializing this network
            weights and biases. Needed for reproducibility.
        :param activation: specifies which activation function to use for the hidden layers
            of the network. 
        """
        
        rng = default_rng(seed)     # needed for reproducibility
        self.training_size = None
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.act = ACTIVATIONS[activation]
        self.momentum = momentum
        self.lmbda = lmbda
        self.last_act = None            # Must be defined by subclassing the Network
        self.g = None
        self.grad_est = None
        self.grad_est_per_epoch = []


        self.biases = [np.zeros_like(l) for l in sizes[1:]]
        self.weights = [rng.normal(0, 1, (y,x))/np.sqrt(x) for x, y in zip(sizes[:-1], sizes[1:])]
        self.wvelocities = [np.zeros_like(weight) for weight in self.weights]
        self.bvelocities = [np.zeros_like(bias) for bias in self.biases]
        self.val_scores = []
        self.train_scores = []

        self.debug = debug


    def feedforward_batch(self, inp):
        out = inp
        units_out = [out]
        nets = []
        for b, w in zip(self.biases[:-1], self.weights[:-1]):
            net = np.matmul(out, w.T) + b
            out = self.act.function(net)
            nets.append(net)
            units_out.append(out)

        # Last layer is linear for regression tasks and sigmoid for classification
        out = self.last_act.function(np.matmul(out, self.weights[-1].T) + self.biases[-1])
        nets.append(out)
        units_out.append(out)

        return units_out, nets, out

    
    def backpropagation_batch(self, x, y, der):
        nabla_b = [0]*(len(self.sizes)-1)
        nabla_w = [0]*(len(self.sizes)-1)

        # Forward computation
        units_out, nets, out = self.feedforward_batch(x)
        delta = 0

        # Backward pass
        for l in range(1, self.num_layers):
            if l == 1:
                # Backward pass - output unit
                delta = (out - y)
                delta = delta * self.last_act.derivative(nets[-1])
            else:
                # Backward pass - hidden unit
                delta = np.matmul(delta, self.weights[-l+1])
                delta = delta * der(nets[-l])

            nabla_b[-l] = delta.sum(axis=0)
            nabla_w[-l] = np.matmul(delta.T, units_out[-l-1])

        return nabla_b, nabla_w


    def train(self, optimizer, training_data, epochs, eta, batch_size=None, test_data=None):
        self.optimizer = OPTIMIZERS[optimizer](training_data, epochs, eta, batch_size=batch_size, test_data=test_data)
        self.optimizer.optimize(self)


    def plot_score(self, name):
        """Utility function, allows to build a plot of the scores achieved during training
        for the validation set and the training set.

        :param name: Prefix name for the file related to the plot.
        """        
        plt.plot(self.val_scores, '--', label='Validation loss')
        plt.plot(self.train_scores, '--', label='Training loss')
        plt.legend(loc='best')
        plt.xlabel ('Epochs')
        plt.ylabel ('Loss')
        plt.title ('Loss NN CUP dataset')
        plt.draw()

        plt.savefig(f"src/NN/res/{name}ep{self.epochs}s{self.sizes}b{self.batch_size}e{self.eta}lmbda{self.lmbda}m{self.momentum}.png")
        plt.clf()

    
    def plot_grad(self, name):
        plt.plot(self.grad_est_per_epoch, '--', label='Validation loss')
        plt.legend(loc='best')
        plt.xlabel ('Epochs')
        plt.ylabel ('Gradient\'s norm')
        plt.title ('Gradient norm estimate')
        plt.draw()

        plt.savefig(f"src/NN/res/grads/{name}ep{self.epochs}s{self.sizes}b{self.batch_size}e{self.eta}lmbda{self.lmbda}m{self.momentum}.png")
        plt.clf()


    @abstractmethod
    def best_score(self):
        """Returns the best score achieved during the fitting of the current network.
        """        
        pass


    @abstractmethod
    def predict(self, data):
        pass


    def evaluate(self, test_data, train_data):
        """Evaluates the performances of the Network in the current state,
        propagating the test examples through the network via a complete feedforward
        step. It evaluates the performance using the R2 metric in order to be
        comparable with sklearn out-of-the-box NN results.

        :param test_data: test data to evaluate the NN
        :return: The R2 score as defined by sklearn library
        """

        score_test = []
        score_train = []
            
        preds_test = self.predict(test_data[0])
        truth_test = test_data[1]

        preds_train = self.predict(train_data[0])
        truth_train = train_data[1]
        
        score_test.append(self.loss.loss(truth_test, preds_test))
        score_train.append(self.loss.loss(truth_train, preds_train))

        self.val_scores.append(score_test)
        self.train_scores.append(score_train)

        return (score_test, score_train), preds_train, preds_test
