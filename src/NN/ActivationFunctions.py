import numpy as np
from abc import ABCMeta, abstractmethod

class ActivationFunction(metaclass=ABCMeta):

    @abstractmethod
    def function(self, x):
        pass

    @abstractmethod
    def derivative(self, x):
        pass


class ReLU(ActivationFunction):
    
    @staticmethod
    def function(x):
        """Utility function that implements the ReLU function used in the NN units.
        It works both with vectors and scalars. 

        :param x: input vector/scalar
        :return: x if x >= 0, 0 otherwise
        """
        return np.maximum(x, 0.)

    @staticmethod
    def derivative(x):
        """Utility function that implements the ReLU function derivative.
        It works both with vectors and scalars.

        :param x: input vector/scalar
        :return: 1 if :param x: is greater than 0, 0 otherwise
        """    
        return 1. * (x > 0)


class Sigmoid(ActivationFunction):

    @staticmethod
    def function(x):
        """Function that defines the sigmoid activation function used in the
        Neaural Network units. It works both with vectors and numbers.

        :param z: element to which apply the sigmoid function.
        :return: an element with the same shape of the input with
                sigmoid function applied elementwise in case of a vector.
        """

        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def derivative(x):
        """Derivative function of the sigmoid function.

        :param z: element to which apply the sigmoid derivative function.
        :return: an element with the same shape of the input with
                sigmoid derivative function applied elementwise in case of a vector.
        """
        x = Sigmoid.function(x)
        return x * (1 - x)


class Linear(ActivationFunction):

    @staticmethod
    def function(x):        
        return x

    @staticmethod
    def derivative(x):
        return 1