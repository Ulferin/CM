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

        # TODO: magari questo si può mettere nelle specifiche indicando che sia train che test devono avere vettore obiettivo come 2d vector
        # Reshape vectors to fit needed shape
        self.training_data = (training_data[0], training_data[1].reshape(training_data[1].shape[0], -1))
        self.test_data = test_data


    @abstractmethod
    def update_mini_batch(self, nn, mini_batch):
        """Updates weights and biases of the specified Neural Network object :nn: by
        using the current mini-batch samples :mini_batch:.

        Parameters
        ----------
        nn : Object
            Neural Network to use for updates and gradient computation.
        mini_batch : np.ndarray
            mini-batch samples to use for updates.
        """              
        pass


    @abstractmethod
    def iteration_end(self, e, nn):
        pass


class SGD(Optimizer):

    def __init__(self, training_data, epochs, eta, batch_size=None, test_data=None):
        super().__init__(training_data, epochs, eta, batch_size=batch_size, test_data=test_data)


    def iteration_end(self, e, nn):
        """Trains the Neural Network :nn: using mini-batch stochastic gradient descent,
        applied to the training examples for the current optimizer for a given
        number of epochs and with the specified learning rate. If test_data exists,
        the learning algorithm will print progresses during the training phase.

        Parameters
        ----------
        nn : Object
            Neural Network to train with the current optimization method.
        """           

        # TODO: vedere cosa fare al termine di ogni iterazione, magari controllare norma gradiente e terminare se raggiunta soluzione
        pass

    def update_mini_batch(self, nn, nabla_b, nabla_w, size):
        """Updates weights and biases of the specified Neural Network object :nn: by
        using the current mini-batch samples :mini_batch:. Uses a regularized momentum
        based approach for the weights update. Hyperparameters must be configured directly
        on the :nn: object.

        Parameters
        ----------
        nn : Object
            Neural Network to use for updates and gradient computation.
        mini_batch : np.ndarray
            mini-batch samples to use for updates.
        """

        # Momentum updates
        nn.wvelocities = [nn.momentum * velocity - (self.eta/size)*nw for velocity,nw in zip(nn.wvelocities, nabla_w)]
        nn.bvelocities = [nn.momentum * velocity - (self.eta/size)*nb for velocity,nb in zip(nn.bvelocities, nabla_b)]

        nn.weights = [w + velocity - (nn.lmbda/size) * w for w,velocity in zip(nn.weights, nn.wvelocities)]
        nn.biases = [b + velocity for b,velocity in zip(nn.biases, nn.bvelocities)]


class SGM(Optimizer):

    def __init__(self, training_data, epochs, eta, eps=1e-5, batch_size=None, test_data=None):
        super().__init__(training_data, epochs, eta, batch_size=batch_size, test_data=test_data)
        self.eps = eps
        self.step = eta
        self.x_ref = []
        self.f_ref = np.inf


    def iteration_end(self, e, nn):
        """Trains the Neural Network :nn: using mini-batch sub-gradient method,
        applied to the training examples for the current optimizer for a given
        number of epochs and with the specified step-size. If test_data exists,
        the learning algorithm will print progresses during the training phase.

        Parameters
        ----------
        nn : Object
            Neural Network to train with the current optimization method.
        """

        self.step = self.eta * (1 / e)

        last_f = nn.score

        # found a better value
        if last_f < self.f_ref:
            self.f_ref = last_f
            self.x_ref = (nn.weights.copy(), nn.biases.copy())

        if nn.ngrad < self.eps:
            print("Reached desired precision.")


    def update_mini_batch(self, nn, nabla_b, nabla_w, size):
        """Updates weights and biases of the specified Neural Network object :nn: by
        using the current mini-batch samples :mini_batch:. Uses a diminishing step-size
        rule for updates.

        Parameters
        ----------
        nn : Object
            Neural Network to use for updates and gradient computation.
        mini_batch : np.ndarray
            mini-batch samples to use for updates.
        """

        # Compute search direction
        d = self.step/nn.ngrad

        nn.weights = [w - d*nw for w,nw in zip(nn.weights, nabla_w)]
        nn.biases = [b - d*nb for b,nb in zip(nn.biases, nabla_b)]