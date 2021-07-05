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
from LossFunctions import MeanSquaredError
from matplotlib import pyplot as plt

from src.NN.ActivationFunctions import ReLU, Sigmoid


# TODO: aggiungere size gradient per ogni step
# TODO: implmentare subgrad
# TODO: usiamo Leaky ReLU e non ReLU

ACTIVATIONS = {
    'relu': ReLU,
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


        self.biases = [np.zeros_like(l) for l in sizes[1:]]
        self.weights = [rng.normal(0, 1, (y,x))/np.sqrt(x) for x, y in zip(sizes[:-1], sizes[1:])]
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
            out = self.act.function(net)
            nets.append(net)
            units_out.append(out)
    
        # Last layer is linear for regression tasks and sigmoid for classification
        out = self.last_act.function(np.matmul(self.weights[-1], out) + self.biases[-1].T)
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
        delta = 0

        # Backward pass
        for l in range(1, self.num_layers):
            if l == 1:
                # Backward pass - output unit
                delta = (out - y.reshape(-1,1))
                delta = delta * self.last_act.derivative(nets[-1])  # TODO: questo si puù cambiare con una sola riga come quella in 118
            else:
                # Backward pass - hidden unit
                delta = np.matmul(self.weights[-l+1].T, delta)
                delta = delta * self.act.derivative(nets[-l])
            
            nabla_b[-l] = delta + (2 * self.lmbda * self.biases[-l].T)     # regularization term derivative
            nabla_w[-l] = np.matmul(delta, units_out[-l-1].T) + (2 * self.lmbda * self.weights[-l]) # regularization term derivative
        
        self.g = delta

        return nabla_b, nabla_w


    def update_mini_batch(self, mini_batch: tuple, eta, sub=False):
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

        if not sub:
            # Momentum updates
            self.wvelocities = [self.momentum * velocity - (eta/len(mini_batch[0]))*nw for velocity,nw in zip(self.wvelocities, nabla_w)]
            self.bvelocities = [self.momentum * velocity - (eta/len(mini_batch[0]))*nb for velocity,nb in zip(self.bvelocities, nabla_b)]

            self.weights = [w + velocity for w,velocity in zip(self.weights, self.wvelocities)]
            self.biases = [b + velocity for b,velocity in zip(self.biases, self.bvelocities)]
        else:
            # Compute search direction
            self.ngrad = np.linalg.norm(np.hstack([el.ravel() for el in nabla_w + nabla_b]))
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

        self.batch_size = batch_size
        self.eta = eta
        self.epochs = epochs
        self.training_size = len(training_data[0])

        n = len(training_data[0])
        rng = default_rng(0)
        rng.shuffle(training_data[0])
        rng = default_rng(0)
        rng.shuffle(training_data[1])
        for e in range(epochs):
            mini_batches = []
            
            if batch_size is not None:
                batches = int(n/batch_size)
                for b in range(batches):
                    start = b * batch_size
                    end = (b+1) * batch_size
                    mini_batches.append((training_data[0][start:end], training_data[1][start:end]))
                mini_batches.append((training_data[0][b*batch_size:], training_data[1][b*batch_size:]))
            else:
                mini_batches.append((training_data[0], training_data[1]))

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

            if test_data is not None:
                score, preds_train, preds_test = self.evaluate(test_data, training_data)
                self.val_scores.append(score[0])
                self.train_scores.append(score[1])
                if self.debug: print(f"pred train: {preds_train[1]} --> target: {training_data[1][1]} || pred test: {preds_test[1]} --> target {test_data[1][1]}")
                print(f"Epoch {e} completed with gradient norm: {np.linalg.norm(self.g)}, {self.g.shape}. Score: {score}")
            else:
                print(f"Epoch {e} completed.")


    def subgrad(self, training_data, start, epochs, test_data=None):
        """Subgradient metdo implementation using a diminishing step size.

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
        
        x_ref = []
        f_ref = np.inf
        curr_iter = 1

        while True:
            self.step = start * (1 / curr_iter)

            
            preds_train = [np.array(self.feedforward(x)[2]).reshape(y.shape) for x,y in zip(training_data[0], training_data[1])]
            truth_train = [y for y in training_data[1]]

            last_f = MeanSquaredError.loss(truth_train, preds_train)
            self.update_mini_batch(training_data, 1, True)

            # found a better value
            if last_f < f_ref:
                f_ref = last_f
                x_ref = (self.weights, self.biases)

            curr_iter += 1
            if curr_iter >= epochs: break

            print(f"{self.ngrad}\t\t{last_f}")


            

    
    def plot_score(self,name):
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
