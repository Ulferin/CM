import numpy as np

def relu(x):
    """Utility function that implements the ReLU function used in the NN units.
    It works both with vectors and scalars. 

    :param x: input vector/scalar
    :return: x if x >= 0, 0 otherwise
    """    
    return np.maximum(x, 0)


def relu_prime(x):
    """Utility function that implements the ReLU function derivative.
    It works both with vectors and scalars.

    :param x: input vector/scalar
    :return: 1 if :param x: is greater than 0, 0 otherwise
    """    
    # TODO: problema quando x è zero!!!
    return np.maximum(x, 0)/x

def ReLU(x):
    return x * (x > 0)

def dReLU(x):
    return 1. * (x > 0)

def sigmoid(z):
    """Function that defines the sigmoid activation function used in the
    Neaural Network units. It works both with vectors and numbers.

    :param z: element to which apply the sigmoid function.
    :return: an element with the same shape of the input with
             sigmoid function applied elementwise in case of a vector.
    """

    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """Derivative function of the sigmoid function.

    :param z: element to which apply the sigmoid derivative function.
    :return: an element with the same shape of the input with
             sigmoid derivative function applied elementwise in case of a vector.
    """

    return sigmoid(z) * (1 - sigmoid(z))