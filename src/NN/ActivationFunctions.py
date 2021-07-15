import numpy as np
from numpy.random import default_rng

from abc import ABCMeta, abstractmethod


class ActivationFunction(metaclass=ABCMeta):

    @abstractmethod
    def function(self, x):
        pass

    @abstractmethod
    def derivative(self, x):
        pass

    @abstractmethod
    def subgrad(self, x):
        pass


class ReLU(ActivationFunction):
    """Class that implements static utility functions related to
    the ReLU activation function."""    
    
    @staticmethod
    def function(x):
        """Implements the ReLU activation function.
        The function is applied elementwise in the case :x: is a vector.
        The returned values, for each element of :x: is:

            · x_i       if x_i >= 0
            · 0         otherwise

        Parameters
        ----------
        x : np.ndarray
            input vector/scalar

        Returns
        -------
        np.ndarray
            Vector with the same shape as input :x: with the ReLU function applied
            elementwise.
        """

        return np.maximum(x, 0.)

    @staticmethod
    def derivative(x):
        """Implements the derivative of the ReLU activation function.
        The derivative is applied elementwise in the case :x: is a vector.
        The returned values, for each element of :x: is:

            · 1         if x_i >= 0
            · 0         otherwise

        Parameters
        ----------
        x : np.ndarray
            Input vector/scalar.

        Returns
        -------
        np.ndarray
            Vector with the same shape as input :x: with the ReLU derivative
            applied elementswise.
        """        

        return 1. * (x >= 0)

    
    @staticmethod
    def subgrad(x):
        """Implements the subgradient of the ReLU activation function.
        The subgradient is applied elementwise in the case :x: is a vector.
        The returned value, for each element of :x: is:
            
            · 1         if x > 0
            · [0,1]     if x = 0    (value selected with a uniform distribution)
            · 0         if x < 0

        Parameters
        ----------
        x : np.ndarray
            Input vector/scalar.

        Returns
        -------
        np.ndarray
            Vector with the same shape as input :x: with the ReLU subgradient
            function applied elementwise.
        """        

        rng = default_rng()
        return np.where(x>0, 1, np.where(x<0, 0, rng.uniform()))


class LeakyReLU(ActivationFunction): 
    """Class that implements static utility functions related to
    the Leaky ReLU activation function.""" 

    @staticmethod
    def function(x):
        """Implements the Leaky ReLU activation function with constant 0.01.
        The function is applied elementwise in the case :x: is a vector.
        The returned values, for each element of :x: is:

            · x_i           if x_i >= 0
            · 0.01 * x_i    otherwise

        Parameters
        ----------
        x : np.ndarray
            input vector/scalar

        Returns
        -------
        np.ndarray
            Vector with the same shape as input :x: with the Leaky ReLU
            function applied elementwise.
        """

        return np.where(x>=0, x, 0.01*x)


    @staticmethod
    def derivative(x):
        """Implements the derivative of the Leaky ReLU activation function.
        The derivative is applied elementwise in the case :x: is a vector.
        The returned values, for each element of :x: is:

            · 1         if x_i >= 0
            · 0.01      otherwise

        Parameters
        ----------
        x : np.ndarray
            Input vector/scalar.

        Returns
        -------
        np.ndarray
            Vector with the same shape as input :x: with the Leaky ReLU
            derivative applied elementwise.
        """

        return np.where(x<0, 0.01, 1)


    # TODO: controllare subgradient di Leaky Relu
    @staticmethod
    def subgrad(x):
        """Implements the subgradient of the Leaky ReLU activation function.
        The subgradient is applied elementwise in the case :x: is a vector.
        The returned value, for each element of :x: is:
            
            · 1         if x > 0
            · [0,1]     if x = 0    (value selected with a uniform distribution)
            · 0.01      if x < 0

        Parameters
        ----------
        x : np.ndarray
            Input vector/scalar.

        Returns
        -------
        np.ndarray
            Vector with the same shape as input :x: with the ReLU subgradient
            function applied elementwise.
        """

        rng = default_rng()
        return np.where(x>0, 1, np.where(x<0, 0.01, rng.uniform()))


class Sigmoid(ActivationFunction):
    """Class that implements static utility functions related to
    the Sigmoid activation function."""

    @staticmethod
    def function(x):
        """Implements the Sigmoid activation function.
        The function is applied elementwise in the case :x: is a vector.
        The returned values, for each element of :x: is:

            · 1.0 / (1.0 + exp(-x_i))

        Parameters
        ----------
        x : np.ndarray
            input vector/scalar

        Returns
        -------
        np.ndarray
            Vector with the same shape as input :x: with the Sigmoid
            function applied elementwise.
        """

        return 1.0 / (1.0 + np.exp(-x))


    @staticmethod
    def derivative(x):
        """Implements the derivative of the Sigmoid activation function.
        The derivative is applied elementwise in the case :x: is a vector.
        The returned values, for each element of :x: is:

            · sigmoid(x_i) * (1 - sigmoid(x_i))

        Parameters
        ----------
        x : np.ndarray
            Input vector/scalar.

        Returns
        -------
        np.ndarray
            Vector with the same shape as input :x: with the Sigmoid
            derivative applied elementwise.
        """

        x = Sigmoid.function(x)
        return x * (1 - x)

    
    @staticmethod
    def subgrad(x):
        """Since Sigmoid activation function is derivable, this is equivalent of
        the derivative function."""

        return Sigmoid.derivative(x)


class Linear(ActivationFunction):
    """Class that implements static utility functions related to
    the Linear activation function."""

    @staticmethod
    def function(x):      
        """Implements the Linear activation function.
        The function is applied elementwise in the case :x: is a vector.
        The returned values, for each element of :x: is:

            · x_i

        Parameters
        ----------
        x : np.ndarray
            input vector/scalar

        Returns
        -------
        np.ndarray
            Vector with the same shape as input :x: with the Linear
            function applied elementwise.
        """

        return x


    @staticmethod
    def derivative(x):
        """Implements the derivative of the Linear activation function.
        The derivative is applied elementwise in the case :x: is a vector.
        The returned values, for each element of :x: is:

            · 1

        Parameters
        ----------
        x : np.ndarray
            Input vector/scalar.

        Returns
        -------
        np.ndarray
            Vector with the same shape as input :x: with the Linear
            derivative applied elementwise.
        """

        return 1

    
    @staticmethod
    def subgrad(x):
        """Since Linear activation function is derivable, this is equivalent of
        the derivative function."""

        return Linear.derivative(x)