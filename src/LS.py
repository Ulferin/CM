import numpy as np


class LS():
    """This class implements the least square problem solver for the given data. It uses the thin QR
    factorization, as described in the report file.
    Solves the problem (P) = min || Ax-b ||
    """

    def solve(self, A, b):
        """Solves the LS problem (P) given the input matrix A and the vector b.

        :param A: Input matrix for the LS problem
        :param b: Input vector for the LS problem
        :return: Result vector that minimizes (P)
        """

        R = self.qr(A)
        _, n = R.shape
        Rinv = np.linalg.inv(R)

        implicit = self.implicit_Qb(b)[:n]
        
        res = np.zeros(n)
        for i in range(n):
            res[i] = (np.dot(Rinv[i],implicit)) 
    
        return res


    def householder_vector(self, x):
        """Computes the householder vector for the given vector :param x:

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
        Q = np.eye(m, n)
        R = A.astype(np.float64)
        u_list = []


        for j in range(np.min((m,n))):   # note that this is always equal to n in our case
            
            s, u = self.householder_vector(R[j:,j])

            # zero division in machine precision
            # this change will cause the matrix R to have 0s in the diagonal
            if np.abs(s) < eps:
                s = 0
                u = np.zeros(len(u))

            u_list.append(u)

            R[j, j] = s
            R[j+1:, j] = 0            

            res = np.zeros(len(R[0])-j-1)
            for i in range(n-j-1):
                res[i] = (np.dot(u, R[j:,i+j+1]))
            R[j:, j+1:] -= 2.*np.outer(u, res)

        self.u_list = u_list
        
        return R[:n, :n]

    
    def implicit_Qb(self, b):
        """Computes implicitly the product Q1*b used in the LS problem solution, starting from the
        Householder vectors computed during the QR factorization phase.

        :param b: Input vector to use in the implicit product.
        :return: The vector Q1*b
        """

        m = len(b)
        b = b.astype(np.float64)
        for k in range(len(self.u_list)):
            b[k:m] -= 2.*np.dot(self.u_list[k], np.multiply(self.u_list[k], b[k:m]))

        return b


    def dot_matvec(self, Q, u, k):
        """Auxiliary function to compute submatrix-vector product starting from the given index on second axis.
        During testing it appeared to show increased performance against np.dot(Q[:,j:],u)

        :param Q: Input matrix.
        :param u: Input vector.
        :param k: Starting index for submatrix.
        :return: Matrix-vector multiplication's result vector
        """

        res = np.zeros(len(Q))
        for i in range(len(Q)):
            res[i] = (np.dot(Q[i][k:],u)) 

        return res


    def revertQ(self):
        """Builds the orthogonal matrix Q resulting from the QR computation, starting from the
        Householder vectors built during the factorization.

        :return: Orthogonal matrix Q.
        """

        n = len(self.u_list)
        m = len(self.u_list[0])

        Q = np.eye(m,m)
        for j in range(n):
            u = self.u_list[j]
            Q[:,j:] -= np.outer(2*self.dot_matvec(Q, u, j), u)

        return Q

        