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
    """Implements static utility functions related to
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
            Vector with the same shape as input :x: with the ReLU function
            applied elementwise.
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


class LeakyReLU(ActivationFunction): 
    """Implements static utility functions related to
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


class Sigmoid(ActivationFunction):
    """Implements static utility functions related to
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

        sgmd = Sigmoid.function(x)
        return sgmd * (1 - sgmd)


class Linear(ActivationFunction):
    """Implements static utility functions related to
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