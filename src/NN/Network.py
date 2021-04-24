# This class represents the Network used in the CM project and it is
# implemented from scratch following the advices taken during the course
# of ML

import numpy as np
from numpy.random import default_rng
from sklearn.metrics import r2_score, mean_squared_error
import random

from functions import relu, relu_prime, ReLU, dReLU, sigmoid, sigmoid_prime

# TODO: check all the list comprehensions, maybe we can substitute them with a numpy method

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
        self.act = sigmoid
        self.der_act = sigmoid_prime

        # TODO: non possiamo avere un singolo bias per layer invece che un bias per ogni unità?
        #       controllare nel libro dove ha dato questo esempio cosa dice a riguardo
        self.biases = [rng.standard_normal((1,y)) for y in sizes[1:]]
        self.weights = [rng.standard_normal((y,x)) for x, y in zip(sizes[:-1], sizes[1:])]
        # NOTE: these are matrices, so also the biases are shaped (1,x)

        self.scores = []


    def feedforward(self, invec):
        """Applies a feedforward pass to the given input :param in:.
        The output generated by the output unit of this network is returned.

        :param in: network input vector
        :return: network output vector (or scalar)
        """
        out = invec.T
        for b, w in zip(self.biases[:-1], self.weights[:-1]):
            out = self.act(np.matmul(w, out) + b.T)

        # Last layer is linear for regression tasks
        return np.matmul(self.weights[-1], out) + self.biases[-1].T


    def backpropagation(self, x, y):
        # NOTE: this is the backpropagation for a single input example!
        # It should perform a feedforward step to compute the current estimated error.
        # After that, it uses the computed error to backpropagate the error participation
        # of each unit. The error participation will then lead to the definition of the delta
        # coefficient used to update the weights and biases for each of the units of the network.
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # Forward computation
        out = x.T
        units_out = [out]
        nets = []
        for b,w in zip(self.biases, self.weights):
            net = np.matmul(w, out) + b.T
            out = self.act(net)
            nets.append(net)
            units_out.append(out)

        # Backward pass - output unit
        delta = (units_out[-1] - y.reshape(-1,1))
        delta = delta * self.der_act(nets[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.matmul(delta, units_out[-2].T)

        # Backward pass - hidden
        for l in range(2, self.num_layers):
            net = nets[-l]
            delta = np.matmul(self.weights[-l+1].T, delta)
            delta = delta * self.der_act(net)
            nabla_b[-l] = delta
            nabla_w[-l] = np.matmul(delta, units_out[-l-1].T)
        
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

        self.weights = [w - (eta/len(mini_batch))*nw for w,nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta/len(mini_batch))*nb for b,nb in zip(self.biases, nabla_b)]


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

        if test_data:
            n_test = len(test_data)

        n = len(training_data)
        for e in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+batch_size] for k in range(0, n, batch_size)
            ]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

            if test_data:
                score = self.evaluate(test_data)
                self.scores.append(score)
                print(f"Epoch {e} completed. Score: {score}")
            else:
                print(f"Epoch {e} completed.")

    
    def best_score(self):
        print(f"The best score for mb:{self.batch_size}, eta: {self.eta}, epochs: {self.epochs}, sizes: {self.sizes} was: {np.min(self.scores)}")


    def evaluate(self, test_data):
        # TODO: generalizzare questo metodo, trovare un modo per specificare come valutare
        #       il risultato. Dividere in base a classification e regression.
        """Evaluates the performances of the Network in the current state,
        propagating the test examples through the network via a complete feedforward
        step. It evaluates the performance using the R2 metric in order to be
        comparable with sklearn out-of-the-box NN results.

        :param test_data: test data to evaluate the NN
        :return: The R2 score as defined by sklearn library
        """        

        preds = [ np.array(self.feedforward(x)).reshape(y.shape) for x,y in test_data]
        truth = [ y for x,y in test_data ]

        for i,(p,t) in enumerate(zip(preds, truth)):
            if i > 10: break
            print(f"p: {p}, t: {t}")

        # score = r2_score(preds, truth)
        score = mean_squared_error(truth, preds)
        return score
