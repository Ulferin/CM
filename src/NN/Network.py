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
from src.NN.LossFunctions import MeanSquaredError
from matplotlib import pyplot as plt

from src.NN.ActivationFunctions import ReLU, Sigmoid, LeakyReLU


# TODO: controllare come effettuare il plotting del gradiente. Prendo la stima corrente data dall'ultimo batch testato?

ACTIVATIONS = {
    'relu': ReLU,
    'Lrelu':LeakyReLU,
    'sigmoid': Sigmoid
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

    
    # TODO: specificare quali sono i vantaggi di usare una singola funzione di attivazione per tutti i layer
    # Batch computation allows us to exploit parallel computation and efficient libraries matrix multiplication
    # given that we are throwing an entire pass over the network as matrix multiplication, except for the last level
    # of the network
    def backpropagation_batch(self, x, y, der):
        nabla_b = [0 for b in self.biases]
        nabla_w = [0 for w in self.weights]

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


    def update_mini_batch(self, mini_batch, eta, der, sub=False):     
        """Updates the network weights and biases by applying the backpropagation algorithm
        to the current set of examples contained in the :mini_batch: param. Computes the deltas
        used to update weights as an average over the size of the examples set, using the provided
        :eta: parameter as learning rate.

        Args:
            mini_batch (tuple): Set of examples to use to update the network weights and biases
            eta (float): Learning rate
            der (function): [description]
            sub (bool, optional): If True, subgradient update is performed. Stochastic one otherwise. Defaults to False.
        """

        nabla_b, nabla_w = self.backpropagation_batch(mini_batch[0], mini_batch[1], der)
        self.ngrad = np.linalg.norm(np.hstack([el.ravel() for el in nabla_w + nabla_b]))

        if not sub:
            # Momentum updates
            self.wvelocities = [self.momentum * velocity - (eta/len(mini_batch[0]))*nw for velocity,nw in zip(self.wvelocities, nabla_w)]
            self.bvelocities = [self.momentum * velocity - (eta/len(mini_batch[0]))*nb for velocity,nb in zip(self.bvelocities, nabla_b)]

            self.weights = [w + velocity - (self.lmbda/len(mini_batch[0]) * w) for w,velocity in zip(self.weights, self.wvelocities)]
            self.biases = [b + velocity for b,velocity in zip(self.biases, self.bvelocities)]
        else:
            # Compute search direction
            dw = self.step/self.ngrad
            db = self.step/self.ngrad

            self.weights = [w - dw*nw for w,nw in zip(self.weights, nabla_w)]
            self.biases = [b - db*nb for b,nb in zip(self.biases, nabla_b)]


    def SGD(self, training_data:tuple, epochs, eta, batch_size=None, test_data:tuple=None):
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

        # Store auxiliary informations to pretty-print statistics
        self.batch_size = batch_size
        self.eta = eta
        self.epochs = epochs
        self.training_size = len(training_data[0])

        # TODO: magari questo si può mettere nelle specifiche indicando che sia train che test devono avere vettore obiettivo come 2d vector
        # Reshape vectors to fit needed shape
        training_data = (training_data[0], training_data[1].reshape(training_data[1].shape[0], 1 if len(training_data[1].shape)==1 else training_data[1].shape[1]))

        rng = default_rng(0)
        rng.shuffle(training_data[0])
        rng = default_rng(0)
        rng.shuffle(training_data[1])

        for e in range(epochs):
            mini_batches = []
            self.grad_est = 0

            if batch_size is not None:
                batches = int(self.training_size/batch_size)
                for b in range(batches):
                    start = b * batch_size
                    end = (b+1) * batch_size
                    mini_batches.append((training_data[0][start:end], training_data[1][start:end]))
                
                # Add remaining data as last batch
                # (it may have different size than previous batches, up to |batch|-1 more elements)
                mini_batches.append((training_data[0][b*batch_size:], training_data[1][b*batch_size:]))
            else:
                mini_batches.append((training_data[0], training_data[1]))

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, self.act.derivative)
                self.grad_est += self.ngrad
            self.grad_est = self.grad_est/self.training_size

            # Compute current gradient estimate
            self.grad_est_per_epoch.append(np.linalg.norm(self.grad_est))

            if test_data is not None:
                score, preds_train, preds_test = self.evaluate(test_data, training_data)
                self.val_scores.append(score[0])
                self.train_scores.append(score[1])
                if self.debug: print(f"pred train: {preds_train[1]} --> target: {training_data[1][1]} || pred test: {preds_test[1]} --> target {test_data[1][1]}")
                print(f"Epoch {e} completed with gradient norm: {self.grad_est}. Score: {score}")
            else:
                print(f"Epoch {e} completed.")


    def subgrad(self, training_data, start, epochs, test_data=None):
        """Subgradient metod implementation using a diminishing step size.

        Parameters
        ----------
        training_data : np.ndarray
            Training samples to use for the training of the current network.
        start : scalar
            starting step to use during the diminishing step size.
        epochs : scalar
            Maximum number of epochs to run this method for.
        test_data : np.ndarray, optional
            Used to evaluate the performances of the network among epochs. By default None.
        """                
        
        self.batch_size = len(training_data[0])
        self.eta = start
        self.epochs = epochs
        self.training_size = len(training_data[0])

        x_ref = []
        f_ref = np.inf
        curr_iter = 1

        while True:
            self.step = start * (1 / curr_iter)
            self.grad_est = 0
            
            preds_train = self.feedforward_batch(training_data[0])[2]
            truth_train = training_data[1]

            last_f = MeanSquaredError.loss(truth_train, preds_train)
            # TODO: per adesso è in versione full batch, magari creare anche qui mini-batch
            self.update_mini_batch(training_data, 1, self.act.subgrad, sub=True)

            # found a better value
            if last_f < f_ref:
                f_ref = last_f
                x_ref = (self.weights, self.biases)

            curr_iter += 1
            if curr_iter >= epochs: break

            if self.debug: print(f"{self.ngrad}\t\t{np.linalg.norm(self.g)}\t\t{last_f}")

            if test_data is not None:
                score, preds_train, preds_test = self.evaluate(test_data, training_data)
                self.val_scores.append(score[0])
                self.train_scores.append(score[1])
                if self.debug: print(f"pred train: {preds_train[1]} --> target: {training_data[1][1]} || pred test: {preds_test[1]} --> target {test_data[1][1]}")
                print(f"Epoch {curr_iter} completed with gradient norm: {self.grad_est/self.training_size}. Score: {score}")
            else:
                print(f"Epoch {curr_iter} completed.")


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
    def evaluate(self, test_data, train_data):
        """Evaluates the performances of the Network in the current state,
        propagating the test examples through the network via a complete feedforward
        step. It evaluates the performance using the R2 metric in order to be
        comparable with sklearn out-of-the-box NN results.

        :param test_data: test data to evaluate the NN
        :return: The R2 score as defined by sklearn library
        """        

        pass
