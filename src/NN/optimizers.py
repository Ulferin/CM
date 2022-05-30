from abc import ABCMeta, abstractmethod
from sklearn.neural_network import MLPRegressor
import numpy as np


class Optimizer(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self, eta, eps=1e-5, lmbda=0.001):
        self.eta = eta
        self.eps = eps
        self.lmbda = lmbda


    @abstractmethod
    def update_parameters(self, parameters, grads):
        pass


    # @abstractmethod
    # def get_updates(self, grads):
    #     pass


    def iteration_end(self, ngrad):
        """Checks if the optimizer has reached an optimal state.

        Parameters
        ----------
        nn : Object
            Neural Network object being trained using this optimizer.

        Returns
        -------
        bool
            Boolean value indicating whether the optimizer has reached an
            optimal state.
        """        

        if ngrad < self.eps:
            return True

        return False



class SGD(Optimizer):

    def __init__(self, eta, eps=1e-5, lmbda=0.01, momentum=0.9, nesterov=True):
        super().__init__(eta, eps=eps, lmbda=lmbda)
        self.nesterov = nesterov
        self.momentum = momentum

        self.v = []

    def update_parameters(self, parameters, grads):
        """Updates weights and biases of the specified Neural Network object
        :nn: by using the current mini-batch samples :mini_batch:. Uses a
        regularized momentum based approach for the weights update.
        Hyperparameters must be configured directly on the :nn: object.

        Parameters
        ----------
        nn : Object
            Neural Network to use for updates and gradient computation.
        
        mini_batch : np.ndarray
            mini-batch samples to use for updates.
        """

        if len(self.v) == 0:
            self.v = [np.zeros_like(param) for param in parameters]

        # Updates velocities with the current momentum coefficient
        self.v = [
            self.momentum * velocity - self.eta*g
            for velocity, g
            in zip(self.v, grads)
        ]

        for param, update in zip((p for p in parameters), self.v):
            param += update
        
        # Nesterov update
        if self.nesterov:
            for param, velocity in zip((p for p in parameters), self.v):
                param += self.momentum * velocity



class Adam(Optimizer):
    def __init__(self, eta, eps=0.00001, beta1=0.9, beta2=0.999, lmbda=0.01):
        super().__init__(eta, eps, lmbda)
        self.beta1 = beta1
        self.beta2 = beta2
        self.t = 0
        self.offset = 1e-8

        self.first_moment = []
        self.second_moment = []


    def update_parameters(self, parameters, grads):
        self.t += 1

        # Initialize/update gradient accumulation variable
        if len(self.first_moment) == 0:
            self.first_moment = [0]*len(grads)
            self.second_moment = [0]*len(grads)

        self.first_moment = [
            self.beta1 * m + (1-self.beta1)*g
            for m, g in zip(self.first_moment, grads)
        ]

        self.second_moment = [
            self.beta2 * v + (1 - self.beta2)*(g ** 2)
            for v, g in zip(self.second_moment, grads)
        ]

        self.learning_rate = (self.eta
            * np.sqrt(1 - self.beta2**self.t)
            / (1 - self.beta1**self.t))

        updates = [
            -self.learning_rate * fm / (np.sqrt(sm) + self.offset)
            for fm, sm in zip(self.first_moment, self.second_moment)]

        for param, update in zip((p for p in parameters), updates):
            param += update
