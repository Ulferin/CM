from QR import QR
import numpy as np
import random
from scipy.spatial import distance
import sys
import time


m = int(sys.argv[1])
n = int(sys.argv[2])

# Generate random matrix for QR decomposition
def generate(m, n):
    M = np.array([ [random.gauss(0,1) for r in range(n)] for c in range(m) ])
    b = np.array([random.gauss(0,1) for r in range(m)])
    return M, b


# task 2: use qr decomp for polynomial regression example
def polyfit(qr, x, y, n):
    return lsqr(qr, x[:, None]**np.arange(n + 1), y.T)


def lsqr(qr, a, b):
    u_list, r = qr.qr(a)
    _, n = r.shape
    implicit = qr.implicit_Qb(b)[:n]
    return np.dot( np.linalg.inv(r), implicit )


def solve(qr, M, b):
    u_list, R = qr.qr(M)
    _, n = R.shape
    implicit = qr.implicit_Qb(b)[:n]
    return u_list, R, np.dot( np.linalg.inv(R), implicit )


def revertQ(u_list, m):
    Q = np.eye(m,m)
    for k in range(len(u_list)):
        u = u_list[k]
        A = np.eye(m,m)
        A[k:,k:] = A[k:,k:] - 2*np.outer(u,u) # u*u' has dimension (m-k+1 x m-k+1) 
        Q = np.dot(Q, A)

    return Q


qr = QR()
 
# x = np.array((0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10))
# y = np.array((1, 6, 17, 34, 57, 86, 121, 162, 209, 262, 321))
# res = polyfit(qr, x, y, 2)
# print("\nreverse: ",np.dot(x[:, None]**np.arange(2 + 1), res))
# print('\npolyfit:\n', res)

M, b = generate(m, n)

# Computes time for LS solver 
startLS = int(round(time.time() * 1000))
u_list, R, res = solve(qr, M, b)
endLS = int(round(time.time() * 1000)) - startLS

# Computes time for Q and QR reconstruction
startQ = int(round(time.time() * 1000))
Q = revertQ(u_list, m)
R_complete = np.zeros((m,n))
R_complete[:n, :n] = R_complete[:n, :n] + R 
QR = np.dot(Q, R_complete)
endQR = int(round(time.time() * 1000)) - startQ

print(f"Solved (m x n): ({m},{n}) in {endLS} msec \
- Reverting and reconstruction in {endQR} msec \
- Matrix error: {np.linalg.norm( QR - M)/np.linalg.norm(M)} \
- QR error: {np.linalg.norm( M - QR )/np.linalg.norm(QR)} \
- L2 distance is: {np.linalg.norm(np.dot(M, res) - b)}\n")
