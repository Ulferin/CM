import numpy as np
from datetime import datetime as dt

class LS():
    """This class implements the least square problem solver for the given data.
    It uses the thin QR factorization, as described in the report file.
    Solves the problem (P) = min || Ax-b ||
    """

    def solve(self, A, b):
        """Solves the LS problem (P) given the input matrix A and the vector b.
        Computes the QR factorization of the coefficient matrix and uses the
        resulting householder vectors to implicitly compute the product Q1*b.
        Solves the problem as x = R^(-1)*Q1*b

        Parameters
        ----------
        A : np.ndarray
            Coefficients matrix for the LS problem
        b : np.ndarray
            Dependend variables vector for the LS problem

        Returns
        -------
        res : np.ndarray
            Result vector that minimizes (P)
        """

        R = self.qr(A)
        _, n = R.shape
        Rinv = np.linalg.inv(R)

        implicit = self.implicit_Qb(b)[:n]
        res = np.matmul(Rinv, implicit)
    
        return res


    def householder_vector(self, x):
        """Computes the householder vector for the given vector :x:.

        Parameters
        ----------
        x : np.ndarray
            Starting vector used to compute the HH vector

        Returns
        -------
        s : float
            Norm of the starting vector :x:
        
        v : np.ndarray
            Vector representing the computed HH vector.
        """        
        
        s = np.linalg.norm(x)
        if x[0] > 0:
            s = -s

        v = np.copy(x)
        v[0] = v[0] - s

        return s, np.divide(v,np.linalg.norm(v))


    def qr(self, A):
        """Computes the QR factorization for the given input matrix :A: with
        dimensions m x n. Please note that this method does not return the
        orthogonal matrix Q resulting from the factorization. It can be
        recovered by using the revert function of this class. 

        Parameters
        ----------
        A : np.ndarray
            Input matrix for which to compute the QR factorization

        Returns
        -------
        R : np.ndarray
            The triangular matrix R for the thin-QR factorization
        """      

        eps = np.linalg.norm(A)/10**16

        m, n = A.shape
        R = A.astype(float)
        u_list = []

        # note that this is always equal to n in our case
        for j in range(np.min((m,n))):
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
        """Computes implicitly the product Q1*b used in the LS problem solution,
        starting from the Householder vectors computed during the QR
        factorization phase.

        Parameters
        ----------
        b : np.ndarray
            Input vector to use in the implicit product.

        Returns
        -------
        b : np.ndarray
            The vector Q1*b
        """

        m = len(b)
        n = len(self.u_list)

        implicit = b.copy()
        for k, u in enumerate(self.u_list):
            implicit[k:m] = implicit[k:m] - u.dot(2*(u.T.dot(implicit[k:m])))

        return implicit


    def implicit_Qx(self, x):
        """Computes the matrix-vector multiplication Q*x implicitly by using
        the computed householder vectors during QR factorization.

        Parameters
        ----------
        x : np.ndarray
            Vector for the implicit matrix-vector product.

        Returns
        -------
        x : np.ndarray
            Result of Q*x
        """        
        m = len(x)
        n = len(self.u_list)

        for k in range(n-1, -1, -1):
                x[k:m] -= self.u_list[k].dot(2*(self.u_list[k].T.dot(x[k:m])) )

        return x


    def revertQ(self):
        """Computes the matrix Q of the QR decomposition by computing the
        product with the column of the identity matrix as shown in
        [Numerical Linear Algebra by Trefethen, Bau - Lecture 10].

        The construction of the matrix Q starts from the Householder vectors
        found during the QR factorization, the resulting Q is here
        reconstructed by using the 'implicit calculation of a product Q*x'.
        The process is repeated for all the columns of the identity matrix
        {I in R^(m x m)} and the result is transposed. This due to a better
        efficiency of numpy in accessing rows rather than columns in a matrix.
    
        Returns
        -------
        Q : np.ndarray
            Orthogonal matrix Q
        """

        n = len(self.u_list)
        m = len(self.u_list[0])

        Q = []
        # we have to form the first n columns of Q
        for i in range(n):
            e_i = np.zeros(m, dtype=float)
            e_i[i] = 1.0
            e_i = self.implicit_Qx(e_i)
            Q.append(e_i)
            
        return (np.array(Q)).T


        