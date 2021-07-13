from abc import ABCMeta, abstractmethod

import numpy as np

from src.NN.LossFunctions import MeanSquaredError

class Optimizer(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self, training_data, epochs, eta, batch_size=None, test_data=None):
        # Store auxiliary informations to pretty-print statistics
        self.batch_size = batch_size
        self.eta = eta
        self.epochs = epochs
        self.training_size = len(training_data[0])
        self.batches = int(self.training_size/batch_size) if batch_size is not None else 1
        self.grad_est_per_epoch = []
        self.val_scores = []
        self.train_scores = []

        # TODO: magari questo si può mettere nelle specifiche indicando che sia train che test devono avere vettore obiettivo come 2d vector
        # Reshape vectors to fit needed shape
        self.training_data = (training_data[0], training_data[1].reshape(training_data[1].shape[0], -1))
        self.test_data = test_data


    def create_batches(self, batches, training_data):
        mini_batches = []

        if self.batch_size is not None:
            for b in range(batches):
                start = b * self.batch_size
                end = (b+1) * self.batch_size
                mini_batches.append((training_data[0][start:end], training_data[1][start:end]))
            
            # Add remaining data as last batch
            # (it may have different size than previous batches, up to |batch|-1 more elements)
            mini_batches.append((training_data[0][b*self.batch_size:], training_data[1][b*self.batch_size:]))
        else:
            mini_batches.append((training_data[0], training_data[1]))

        return mini_batches


    def evaluate(self, e, nn):
        if self.test_data is not None:
            score, preds_train, preds_test = nn.evaluate(self.test_data, self.training_data)
            if nn.debug: print(f"pred train: {preds_train[1]} --> target: {self.training_data[1][1]} || pred test: {preds_test[1]} --> target {self.test_data[1][1]}")
            print(f"Epoch {e}. Gradient norm: {self.grad_est}. Score: {score}")
        else:
            print(f"Epoch {e} completed.")


    def __update_mini_batch__(self, nn, mini_batch, der):     
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

        nabla_b, nabla_w = nn.backpropagation_batch(mini_batch[0], mini_batch[1], der)
        self.ngrad = np.linalg.norm(np.hstack([el.ravel() for el in nabla_w + nabla_b]))

        return nabla_b, nabla_w


    @abstractmethod
    def update_mini_batch(self, nn, mini_batch):
        pass


    def update_batches(self, nn):
        mini_batches = self.create_batches(self.batches, self.training_data)
        self.grad_est = 0

        for mini_batch in mini_batches:
            self.update_mini_batch(nn, mini_batch)
            self.grad_est += self.ngrad
        self.grad_est = self.grad_est/self.batches


    @abstractmethod
    def optimize(self, nn):
        pass


class SGD(Optimizer):

    def __init__(self, training_data, epochs, eta, batch_size=None, test_data=None):
        super().__init__(training_data, epochs, eta, batch_size=batch_size, test_data=test_data)


    def optimize(self, nn):
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
        for e in range(self.epochs):
            self.update_batches(nn)

            # Compute current gradient estimate
            self.grad_est_per_epoch.append(self.grad_est)
            self.evaluate(e, nn)


    def update_mini_batch(self, nn, mini_batch):
        nabla_b, nabla_w = self.__update_mini_batch__(nn, mini_batch, nn.act.derivative)

        # Momentum updates
        nn.wvelocities = [nn.momentum * velocity - (self.eta/len(mini_batch[0]))*nw for velocity,nw in zip(nn.wvelocities, nabla_w)]
        nn.bvelocities = [nn.momentum * velocity - (self.eta/len(mini_batch[0]))*nb for velocity,nb in zip(nn.bvelocities, nabla_b)]

        nn.weights = [w + velocity - (nn.lmbda/len(mini_batch[0]) * w) for w,velocity in zip(nn.weights, nn.wvelocities)]
        nn.biases = [b + velocity for b,velocity in zip(nn.biases, nn.bvelocities)]


class SGM(Optimizer):

    def __init__(self, training_data, epochs, eta, batch_size=None, test_data=None):
        super().__init__(training_data, epochs, eta, batch_size=batch_size, test_data=test_data)


    def optimize(self, nn):
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

        x_ref = []
        f_ref = np.inf

        for e in range(1, self.epochs):
            self.step = self.eta * (1 / e)
            
            preds_train = nn.feedforward_batch(self.training_data[0])[2]
            truth_train = self.training_data[1]

            last_f = MeanSquaredError.loss(truth_train, preds_train)

            self.update_batches(nn)

            # found a better value
            if last_f < f_ref:
                f_ref = last_f
                x_ref = (nn.weights, nn.biases)

            self.evaluate(e, nn)


    def update_mini_batch(self, nn, mini_batch):
        nabla_b, nabla_w = self.__update_mini_batch__(nn, mini_batch, nn.act.subgrad)

        # Compute search direction
        d = self.eta/self.ngrad

        nn.weights = [w - (d/len(mini_batch[0]))*nw for w,nw in zip(nn.weights, nabla_w)]
        nn.biases = [b - (d/len(mini_batch[0]))*nb for b,nb in zip(nn.biases, nabla_b)]