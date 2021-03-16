import numpy as np


class QR():
    """This class implements the classic 'thin QR' factorization used to solve the least square
    problem as described in the CM report.
    """
    def __init__(self):
        self.R = None
        self.u_list = None


    def householder_vector(self, x):
        """Computes the householder vector for the given vector :param x:

        :param x: Starting vector used to compute the HH vector
        :returns: Vector representing the computed HH vector
        """
        
        s = np.linalg.norm(x)
        if x[0] > 0:
            s = -s

        v = np.copy(x)
        v[0] = v[0] - s

        return s, (v/np.linalg.norm(v))


    def qr(self, A):
        """Computes the QR factorization for the given input matrix :param A: with dimensions m x n

        :param A: Input matrix for which to computer the QR factorization
        :returns: The triangular matrix R and a list of householder vectors used to compute the QR factorization
        """

        m, n = A.shape
        Q = np.eye(m, n)
        R = np.copy(A).astype(np.float64)
        u_list = []

        for j in range(np.min((m,n))):
            s, u = self.householder_vector(R[j:,j])
            u_list.append(u)

            R[j, j] = s
            R[j+1:, j] = 0
            first = np.dot(u, R[j:, j+1:])
            second = 2.*np.outer(u, (first))
            R[j:, j+1:] = -(second - R[j:, j+1:])

        self.u_list = u_list
        self.R = R
        # TODO: non serve restituire la u_list
        return u_list, R[:np.min((m,n)), :np.min((m,n))]

    
    def implicit_Qb(self, b):
        if self.u_list == None:
            print("You must first compute the QR factorization!")
            return

        m = len(b)
        b = b.astype(np.float64)
        for k in range(len(self.u_list)):
            b[k:m] = b[k:m] - 2.*np.dot(self.u_list[k], np.dot(self.u_list[k], b[k:m]))

        return b

        