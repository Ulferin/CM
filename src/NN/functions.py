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
    return np.maximum(x, 0)/x