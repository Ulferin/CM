import numpy as np

from abc import ABCMeta, abstractmethod


class ActivationFunction(metaclass=ABCMeta):
    """Abstract base class for the activation functions used in the implemented
    neural network. Provides the interface for the derived classes implementing
    specific activation functions. Each derived class must implement the methods
    needed to compute the function value and the derivative of the function,
    needed in the backpropagation phase.
    """    

    @abstractmethod
    def function(self, x):
        pass

    @abstractmethod
    def derivative(self, x):
        pass


class ReLU(ActivationFunction):
    """Implements static functionalities related to the ReLU activation function.
    """    
    
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
            Input vector/scalar over which the activation function is applied.

        Returns
        -------
          : np.ndarray
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

        The ReLU function is not differentiable at zero, however, as suggested
        by [Deep Learning. Goodfellow, Bengio, Courville. 2016] the derivative
        at zero can be set to 1.

        Parameters
        ----------
        x : np.ndarray
            Input vector/scalar over which the activation function derivative
            is applied.

        Returns
        -------
          : np.ndarray
            Vector with the same shape as input :x: with the ReLU derivative
            applied elementswise.
        """        

        return 1. * (x >= 0)


class LeakyReLU(ActivationFunction): 
    """Implements static functionalities related to the Leaky ReLU activation
    function. This kind of activation function is used to reduce dying ReLU
    problem associated with the ReLU activation function. 
    """

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
            Input vector/scalar over which the activation function is applied.

        Returns
        -------
          : np.ndarray
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

        As for the ReLU activation function, also the Leaky ReLU function is not
        differentiable at zero, however, also in this case we use the convention
        of defining the derivative at zero to be 1.

        Parameters
        ----------
        x : np.ndarray
            Input vector/scalar over which the activation function derivative
            is applied.

        Returns
        -------
          : np.ndarray
            Vector with the same shape as input :x: with the Leaky ReLU
            derivative applied elementwise.
        """

        return np.where(x<0, 0.01, 1)


class Sigmoid(ActivationFunction):
    """Implements static functionalities related to the sigmoid activation
    function. This is the activation function which is used in the final version
    of the project for the CM course both for internal units and output units
    in case of binary classification tasks.

    To identify the predicted class, the sigmoid function's output is checked
    and for values which are above 0.5, the class is considered to be 1,
    otherwise 0.
    """

    @staticmethod
    def function(x):
        """Implements the Sigmoid activation function.
        The function is applied elementwise in the case :x: is a vector.
        The returned values, for each element of :x: is:

            · 1.0 / (1.0 + exp(-x_i))

        The function is continuously differentiable and always returns values
        between 0 and 1.

        Parameters
        ----------
        x : np.ndarray
            Input vector/scalar over which the activation function is applied.

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
            Input vector/scalar over which the activation function derivative
            is applied.

        Returns
        -------
        np.ndarray
            Vector with the same shape as input :x: with the Sigmoid
            derivative applied elementwise.
        """

        sgmd = Sigmoid.function(x)
        return sgmd * (1 - sgmd)


class Linear(ActivationFunction):
    """Implements static functionalities related to the linear activation
    function. This is the activation function which is used in the output layer
    of networks used to solve regression tasks.

    The output of the linear activation function is taken 'as is' to compute
    the predictions of the network with respect to it's inputs. 
    """

    @staticmethod
    def function(x):      
        """Implements the Linear activation function.
        The function simply returns the same values contained in :x:.
        The returned values, for each element of :x: is:

            · x_i

        Parameters
        ----------
        x : np.ndarray
            Input vector/scalar over which the activation function is applied.

        Returns
        -------
         : np.ndarray
            Vector with the same shape as input :x: with the linear
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
            Input vector/scalar over which the activation function derivative
            is applied.

        Returns
        -------
         : np.ndarray
            Vector with the same shape as input :x: with the Linear
            derivative applied elementwise.
        """

        return 1