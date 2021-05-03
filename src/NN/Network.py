# This class represents the Network used in the CM project and it is
# implemented from scratch following the advices taken during the course
# of ML

from abc import ABCMeta, abstractmethod
import numpy as np
from numpy.linalg import norm
from numpy.random import default_rng
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from matplotlib import pyplot as plt

import time
import random

from functions import relu, relu_prime, ReLU, dReLU, sigmoid, sigmoid_prime


# TODO: la mean squared error nella evaluate va implementata da me!!! Non posso usare quella di sklearn

ACTIVATIONS = {
    'relu': [relu, relu_prime],
    'sigmoid': [sigmoid, sigmoid_prime],
    'relu2': [ReLU, dReLU]
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
        
        rng = default_rng(seed)
        self.training_size = None
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.act = ACTIVATIONS[activation][0]
        self.der_act = ACTIVATIONS[activation][1]
        self.momentum = momentum
        self.lmbda = lmbda
        self.last_act = None            # Must be defined by subclassing the Network
        self.last_der = None            # Must be defined by subclassing the Network

        # TODO: non possiamo avere un singolo bias per layer invece che un bias per ogni unità?
        #       controllare nel libro dove ha dato questo esempio cosa dice a riguardo
        # self.biases = [rng.normal(0, 0.01, (1,y)) for y in sizes[1:]]
        self.biases = [np.zeros_like(y) for y in sizes[1:]]
        self.weights = [rng.normal(0, 0.01, (y,x))/np.sqrt(x) for x, y in zip(sizes[:-1], sizes[1:])]
        self.wvelocities = [np.zeros_like(weight) for weight in self.weights]
        self.bvelocities = [np.zeros_like(bias) for bias in self.biases]
        self.val_scores = []
        self.train_scores = []

        self.debug = debug



    def feedforward(self, invec):
        """Applies a feedforward pass to the given input :param in:.
        The output generated by the output unit of this network is returned.

        :param in: network input vector
        :return: network output vector (or scalar)
        """
        out = invec.T
        units_out = [out]
        nets = []
        for b, w in zip(self.biases[:-1], self.weights[:-1]):
            net = np.matmul(w, out) + b.T
            out = self.act(net)
            nets.append(net)
            units_out.append(out)
    
        # Last layer is linear for regression tasks and sigmoid for classification
        out = self.last_act(np.matmul(self.weights[-1], out) + self.biases[-1].T)
        nets.append(out)
        units_out.append(out)
        
        return units_out, nets, out


    def backpropagation(self, x, y):
        """Performs a backpropagation step for the given input sample. It runs a forward
        step to compute the current output and error. It then uses the error to compute
        the contribution of each network unit to the final error. It finally updates weights
        and biases related to the computed contribution.

        :param x: the current test sample for which to compute the error
        :param y: the expected output for the given sample
        :return: the set of updates to be performed for each unit
        """        

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # Forward computation
        units_out, nets, out = self.feedforward(x)

        # Backward pass - output unit
        delta = (out - y.reshape(-1,1))
        delta = delta * self.last_der(nets[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.matmul(delta, units_out[-2].T)

        # Backward pass - hidden
        for l in range(2, self.num_layers):
            net = nets[-l]
            delta = np.matmul(self.weights[-l+1].T, delta)
            delta = delta * self.der_act(net)
            
            if self.lmbda > 0:
                nabla_b[-l] = delta
                nabla_w[-l] = np.matmul(delta, units_out[-l-1].T)
            else:
                nabla_b[-l] = delta + (2 * self.lmbda * self.biases[-l].T)     # regularization term derivative
                nabla_w[-l] = np.matmul(delta, units_out[-l-1].T) + (2 * self.lmbda * self.weights[-l]) # regularization term derivative
        
        return nabla_b, nabla_w


    def update_mini_batch(self, mini_batch, eta):
        """Updates the network weights and biases by applying the backpropagation algorithm
        to the current set of examples contained in the :param mini_batch:. Computes the deltas
        used to update weights as an average over the size of the examples set, using the provided
        :param eta: as learning rate.

        :param mini_batch: Set of examples to use to update the network weights and biases
        :param eta: Learning rate
        """
        nabla_b = [ np.zeros(b.shape) for b in self.biases ]
        nabla_w = [ np.zeros(w.shape) for w in self.weights ]

        for x, y in mini_batch:
            delta_b, delta_w = self.backpropagation(x,y)
            nabla_b = [ nb + db.T for nb,db in zip(nabla_b, delta_b)]
            nabla_w = [ nw + dw for nw,dw in zip(nabla_w, delta_w) ]

        # Momentum updates
        self.wvelocities = [self.momentum * velocity - (eta/len(mini_batch))*nw for velocity,nw in zip(self.wvelocities, nabla_w)]
        self.bvelocities = [self.momentum * velocity - (eta/len(mini_batch))*nb for velocity,nb in zip(self.bvelocities, nabla_b)]

        self.weights = [w + velocity for w,velocity in zip(self.weights, self.wvelocities)]
        self.biases = [b + velocity for b,velocity in zip(self.biases, self.bvelocities)]

    def update_mini_batch_tup(self, mini_batch: tuple, eta):
        """Updates the network weights and biases by applying the backpropagation algorithm
        to the current set of examples contained in the :param mini_batch:. Computes the deltas
        used to update weights as an average over the size of the examples set, using the provided
        :param eta: as learning rate.

        :param mini_batch: Set of examples to use to update the network weights and biases
        :param eta: Learning rate
        """
        nabla_b = [ np.zeros(b.shape) for b in self.biases ]
        nabla_w = [ np.zeros(w.shape) for w in self.weights ]

        for x, y in zip(mini_batch[0], mini_batch[1]):
            delta_b, delta_w = self.backpropagation(x,y)
            nabla_b = [ nb + db.T for nb,db in zip(nabla_b, delta_b)]
            nabla_w = [ nw + dw for nw,dw in zip(nabla_w, delta_w) ]

        # Momentum updates
        self.wvelocities = [self.momentum * velocity - (eta/len(mini_batch))*nw for velocity,nw in zip(self.wvelocities, nabla_w)]
        self.bvelocities = [self.momentum * velocity - (eta/len(mini_batch))*nb for velocity,nb in zip(self.bvelocities, nabla_b)]

        self.weights = [w + velocity for w,velocity in zip(self.weights, self.wvelocities)]
        self.biases = [b + velocity for b,velocity in zip(self.biases, self.bvelocities)]


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

        self.batch_size = batch_size
        self.eta = eta
        self.epochs = epochs
        self.training_size = len(training_data)

        # training_data = np.array(training_data, dtype=object)
        # rng = default_rng()
        # rng.shuffle(training_data)
        n = len(training_data)
        for e in range(epochs):
            mini_batches = [
                training_data[k:k+batch_size] for k in range(0, n, batch_size)
            ]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

            if test_data:
                score = self.evaluate(test_data, training_data)
                self.val_scores.append(score[0])
                self.train_scores.append(score[1])
                if self.debug: print(f"Epoch {e} completed. Score: {score}")
            else:
                print(f"Epoch {e} completed.")


    def SGD_tup(self, training_data:tuple, epochs, batch_size, eta, test_data:tuple=None):
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

        self.batch_size = batch_size
        self.eta = eta
        self.epochs = epochs
        self.training_size = len(training_data[0])

        n = len(training_data[0])
        # rng = default_rng(0)
        # rng.shuffle(training_data[0])
        # rng = default_rng(0)
        # rng.shuffle(training_data[1])
        for e in range(epochs):
            mini_batches = [
                (training_data[0][k:k+batch_size], training_data[1][k:k+batch_size]) for k in range(0, n, batch_size)
            ]

            for mini_batch in mini_batches:
                self.update_mini_batch_tup(mini_batch, eta)

            if test_data:
                score = self.evaluate_tup(test_data, training_data)
                self.val_scores.append(score[0])
                self.train_scores.append(score[1])
                if self.debug: print(f"Epoch {e} completed. Score: {score}")
            else:
                print(f"Epoch {e} completed.")

    
    def plot_score(self,name):
        """Utility function, allows to build a plot of the scores achieved during training
        for the validation set and the training set.

        :param name: Prefix name for the file related to the plot.
        """        
        plt.plot(self.val_scores, '--', label='Validation loss')
        plt.plot(self.train_scores, '--', label='Training loss')
        plt.legend(loc='upper right')
        plt.xlabel ('Epochs')
        plt.ylabel ('Loss')
        plt.title ('Loss NN CUP dataset')
        plt.draw()

        plt.savefig(f"./res/{name}ep{self.epochs}e{self.eta}b{self.batch_size}m{self.momentum}lmbda{self.lmbda}s{self.sizes}.png")
        plt.clf()


    @abstractmethod
    def best_score(self):
        """Returns the best score achieved during the fitting of the current network.
        """        
        pass


    @abstractmethod
    def evaluate(self, test_data, train_data):
        """Evaluates the performances of the Network in the current state,
        propagating the test examples through the network via a complete feedforward
        step. It evaluates the performance using the R2 metric in order to be
        comparable with sklearn out-of-the-box NN results.

        :param test_data: test data to evaluate the NN
        :return: The R2 score as defined by sklearn library
        """        

        pass

    @abstractmethod
    def evaluate_tup(self, test_data, train_data):
        """Evaluates the performances of the Network in the current state,
        propagating the test examples through the network via a complete feedforward
        step. It evaluates the performance using the R2 metric in order to be
        comparable with sklearn out-of-the-box NN results.

        :param test_data: test data to evaluate the NN
        :return: The R2 score as defined by sklearn library
        """        

        pass
