from QR import QR
import numpy as np
import random
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
startQR = int(round(time.time() * 1000))
_, R = qr.qr(M)
Q = qr.revertQ()
R_complete = np.zeros((m,n))
R_complete[:n, :n] = R_complete[:n, :n] + R
QR = np.dot(Q, R_complete)
endQR = int(round(time.time() * 1000)) - startQR


# Computes QR factorization using numpy
startQRnp = int(round(time.time() * 1000))
Qnp, Rnp = np.linalg.qr(M, mode="complete")
QRnp = np.dot(Qnp, Rnp)
endQRnp = int(round(time.time() * 1000)) - startQRnp

# Computes time for LS solver using numpy
startLSnp = int(round(time.time() * 1000))
np.linalg.lstsq(M,b,rcond=-1)
endLSnp = int(round(time.time() * 1000)) - startLSnp


print(f"Solved (m x n): ({m},{n}) in {endLS} msec, w/ np in {endLSnp} msec \
- Reverting and reconstruction: {endQR} msec, w/ np took: {endQRnp} msec")
print(f"QR error: {np.linalg.norm( M - QR )/np.linalg.norm(QR)} \
- QR error w/ np: {np.linalg.norm( M - QRnp )/np.linalg.norm(QRnp)}\n")
