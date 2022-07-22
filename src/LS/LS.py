import numpy as np

class LS():
    """This class implements the least square solver using the thin QR
    factorization method as described in the report file. The class can be used
    to solve the LS problem (P) = min || A@x - b || and to compute the QR
    factorization of the coefficient matrix for the problem (P).

    The solution of problem (P) is computed as follows:
        1. Compute the QR factorization of the coefficient matrix A by using
           the householder vectors.
        2. Compute the implicit product Q1*b
        3. Solve the problem as x = R^(-1)*Q1*b

    In the solution, there is no need to compute the explicit matrix Q, however
    the class offers the option to compute the orthogonal matrix Q deriving it
    from the computed householder vectors. The reconstructed matrix Q can be
    used to compare the accuracy of the QR factorization used by the LS solver.
    """

    def solve(self, A, b):
        """Solves the LS problem (P) given the input matrix A and the vector b.
        Computes the QR factorization of the coefficient matrix and uses the
        resulting householder vectors to implicitly compute the product Q1*b.
        Solves the problem as x = R^(-1)*Q1*b

        Parameters
        ----------
        A : np.ndarray
            Coefficients matrix for the LS problem.
        b : np.ndarray
            Dependend variables vector for the LS problem.

        Returns
        -------
        res : np.ndarray
            Result vector that minimizes (P) in 2-norm.
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
            Starting vector used to compute the HH vector.

        Returns
        -------
        s : float
            Norm of the starting vector :x:.
        
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
        """Computes the thin QR factorization for the given input matrix :A: with
        dimensions m x n. This method does not return the orthogonal matrix Q
        resulting from the factorization. It can be recovered by using the
        revert function of this class. 

        Parameters
        ----------
        A : np.ndarray
            Input matrix for which to compute the QR factorization.

        Returns
        -------
        R : np.ndarray
            The triangular matrix R for the thin-QR factorization.
        """      

        m, n = A.shape
        R = A.astype(float)
        u_list = []

        # note that this is always equal to n in our case
        for j in range(np.min((m,n))):
            s, u = self.householder_vector(R[j:,j])

            u_list.append(u.reshape(-1,1))

            R[j, j] = s
            R[j+1:, j] = 0
            R[j:, j+1:] -= np.outer(u, np.matmul(2*u.T, R[j:, j+1:]))

        self.u_list = u_list
        
        return R[:n, :n]

    
    def implicit_Qb(self, b):        
        """Computes implicitly the product Q1*b used in the LS problem solution,
        starting from the Householder vectors computed during the QR
        factorization phase, it follows the procedure shown in
        [Numerical Linear Algebra by Trefethen, Bau - Lecture 10].

        Parameters
        ----------
        b : np.ndarray
            Input vector to use in the implicit product.

        Returns
        -------
        implicit : np.ndarray
            The implicit computation of the Q1*b quantity.
        """

        m = len(b)
        n = len(self.u_list)

        implicit = b.copy()
        for k, u in enumerate(self.u_list):
            implicit[k:m] = implicit[k:m] - u.dot(2*(u.T.dot(implicit[k:m])))

        return implicit


    def implicit_Qx(self, x):
        """Computes the matrix-vector multiplication Q*x implicitly by using
        the computed householder vectors during QR factorization. Implemented
        following [Numerical Linear Algebra by Trefethen, Bau - Lecture 10].
        Used to recover the matrix Q from the householder vectors computed during
        the QR factorization phase.

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

        The matrix Q is reconstructed starting from the householder vectors
        computed during the QR factorization phase and the columns of the 
        identity matrix. By using the implicit product Q1*x,
        the matrix Q can be reconstructed even after the QR phase is completed.
    
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


        