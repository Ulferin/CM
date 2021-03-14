import numpy as np

class QR():

    def __init__(self):
        pass


    def householder_vector(self, x):
        """Computes the householder vector for the given vector :param x:

        :param x: Starting vector used to compute the HH vector
        :returns: Vector representing the computed HH vector
        """
        
        s = np.linalg.norm(x)
        if x[0] > 0:
            s *= -1

        v = np.copy(x)
        v[0] = v[0] - s

        return s, (v/np.linalg.norm(v))  # pay attention here, it returns a numpy array


    def qr(self, A):
        """Computes the QR factorization for the given input matrix :param A: with dimensions m x n

        :param A: Input matrix for which to computer the QR factorization
        :returns: The triangular matrix R and a list of householder vectors used to compute the QR factorization
        """

        m, n = A.shape
        Q = np.eye(m, n)
        R = np.copy(A)
        u_list = []

        for j in range(np.min((m,n))):
            s, u = self.householder_vector(R[j:,j])
            u_list.append(u)

            R[j, j] = s
            R[j+1:, j] = 0
            first = np.dot(u, R[j:, j+1:])
            second = 2*np.outer(u, (first))
            # print(f"first: {first} second")
            R[j:, j+1:] = R[j:, j+1:] - second
            # Q[:, j:end] = Q[:, j:end] - Q[:, j:end]*u*2*np.transpose(u)  # Actually there is no need to return this matrix in the solving of LSP

        return u_list, R