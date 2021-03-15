from QR import QR
import numpy as np
import random
from scipy.spatial import distance
import sys

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
    return np.dot( np.linalg.inv(R), implicit )

qr = QR()
 
# x = np.array((0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10))
# y = np.array((1, 6, 17, 34, 57, 86, 121, 162, 209, 262, 321))
# res = polyfit(qr, x, y, 2)
# print("\nreverse: ",np.dot(x[:, None]**np.arange(2 + 1), res))
# print('\npolyfit:\n', res)

M, b = generate(m, n)
res = solve(qr, M, b)
print(f"Solved (m x n): ({m},{n}) - Cosine distance is: {distance.cosine(np.dot(M, res), b)} - L2 distance is: {np.linalg.norm(np.dot(M, res) - b)}")
print("Done!")
