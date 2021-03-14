    """This file represents the implementation of the direct solver discussed in the report
    for the CM project. It contains the class used to solve the QR factorization.
    """

import numpy as np

class QR:

    def __init__(self):
        pass


    def householder_vector(self, x):
        """Computes the householder vector for the given vector :param x:

        :param x: Starting vector used to compute the HH vector
        :returns: Vector representing the computed HH vector
        """
        
        n = np.linalg.norm(x)
        if x[0] > 0:
            n *= -1

        v = x
        v[0] = v[0] - s

        return (v/np.linalg.norm(v))  # pay attention here, it returns a numpy array
