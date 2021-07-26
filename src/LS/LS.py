import numpy as np
from datetime import datetime as dt

# TODO: controllare il fatto che inversa in calcolo soluzione sia fattibile, altrimenti trovare altro
# TODO: controllo errori
# TODO: check rank deficiency
# TODO: check this for performance improvement: https://shihchinw.github.io/2019/03/performance-tips-of-numpy-ndarray.html

class LS():
    """This class implements the least square problem solver for the given data. It uses the thin QR
    factorization, as described in the report file.
    Solves the problem (P) = min || Ax-b ||
    """

    def solve(self, A, b):
        """Solves the LS problem (P) given the input matrix A and the vector b.
        Computes the QR factorization of the coefficient matrix and uses the resulting householder
        vectors to implicitly compute the product Q1*b.
        Solves the problem as x = R^(-1)*Q1*b

        :param A: Coefficients matrix for the LS problem
        :param b: Dependend variables vector for the LS problem
        :return: Result vector that minimizes (P)
        """

        R = self.qr(A)
        _, n = R.shape
        Rinv = np.linalg.inv(R)

        implicit = self.implicit_Qb(b)[:n]
        res = np.matmul(Rinv, implicit)
    
        return res


    def householder_vector(self, x):
        """Computes the householder vector for the given vector :param x:.

        :param x: Starting vector used to compute the HH vector
        :returns: Vector representing the computed HH vector and norm of :param x:
        """
        
        s = np.linalg.norm(x)
        if x[0] > 0:
            s = -s

        v = np.copy(x)
        v[0] = v[0] - s

        return s, np.divide(v,np.linalg.norm(v))


    def qr(self, A):
        """Computes the QR factorization for the given input matrix :param A: with dimensions m x n
        Please not that this method does not return the orthogonal matrix Q resulting for the factorization.
        It can be recovered by using the revert function of this class. 

        :param A: Input matrix for which to computer the QR factorization
        :returns: The triangular matrix R for the thin-QR factorization
        """

        eps = np.linalg.norm(A)/10**16

        m, n = A.shape
        R = A.astype(np.single)
        u_list = []

        for j in range(np.min((m,n))):   # note that this is always equal to n in our case
            s, u = self.householder_vector(R[j:,j])

            # zero division in machine precision
            # this change will cause the matrix R to have 0s in the diagonal
            if np.abs(s) < eps:
                s = 0
                u = np.zeros(len(u))

            u_list.append(u.reshape(-1,1))

            R[j, j] = s
            R[j+1:, j] = 0
            R[j:, j+1:] -= np.outer(u, np.matmul(2*u.T, R[j:, j+1:]))

        self.u_list = u_list
        
        return R[:n, :n]

    
    def implicit_Qb(self, b):
        """Computes implicitly the product Q1*b used in the LS problem solution, starting from the
        Householder vectors computed during the QR factorization phase.

        :param b: Input vector to use in the implicit product.
        :return: The vector Q1*b
        """

        m = len(b)
        if b.dtype != np.single:
            b = b.astype(np.single)
            
        for k, u in enumerate(self.u_list):
            b[k:m] -= 2*np.matmul(u, np.matmul(u.T, b[k:m]))

        return b


    def implicit_Qx(self, e_i):
        m = len(e_i)
        n = len(self.u_list)

        for k in range(n-1, -1, -1):
                e_i[k:m] -= np.matmul( self.u_list[k], np.matmul(self.u_list[k].T, 2*e_i[k:m]) )

        return e_i


    def revertQ(self):
        """Computes the matrix Q of the QR decomposition by computing the product with the column
        of the identity matrix as shown in [Numerical Linear Algebra by Trefethen, Bau - Lecture 10].

        The construction of the matrix Q starts from the Householder vectors found during the QR factorization,
        the resulting Q is here reconstructed by using the 'implicit calculation of a product Q*x'.
        The process is repeated for all the columns of the identity matrix {I in R^(m x m)} and the result is
        transposed. This due to a better efficiency of numpy in accessing rows rather than columns in a matrix.
        
        :return: Orthogonal matrix Q
        """

        n = len(self.u_list)
        m = len(self.u_list[0])

        Q = []
        # we have to form the first n columns of Q
        for i in range(n):
            e_i = np.zeros(m, dtype=np.single)
            e_i[i] = 1.0
            e_i = self.implicit_Qx(e_i)
            Q.append(e_i)
            
        return (np.array(Q)).T


        