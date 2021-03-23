from LS import LS
import numpy as np
import random
import sys
import time

from datetime import datetime as dt


m = int(sys.argv[1])
n = int(sys.argv[2])

# Generate random matrix for QR decomposition
def generate(m, n):
    M = np.array([ [random.gauss(0,1) for r in range(n)] for c in range(m) ])
    b = np.array([random.gauss(0,1) for r in range(m)])
    return M, b


def end_time(start):
    end = (dt.now() - start)
    return end.seconds * 1000 + end.microseconds / 1000

ls = LS()
M, b = generate(m, n)

# Computes time for LS solver 
startLS = dt.now()
res = ls.solve(M,b)
endLS = end_time(startLS)

# Computes time for Q and QR reconstruction
R = ls.qr(M)
startQR = dt.now()
Q = ls.revertQ()
# Q = ls.efficient_revert()
R_complete = np.zeros((m,n))
R_complete[:n, :n] = R_complete[:n, :n] + R
QR = np.dot(Q, R_complete)
endQR = end_time(startQR)


# Computes QR factorization using numpy
startQRnp = dt.now()
Qnp, Rnp = np.linalg.qr(M, mode="complete")
QRnp = np.dot(Qnp, Rnp)
endQRnp = end_time(startQRnp)

# Computes time for LS solver using numpy
startLSnp = dt.now()
np.linalg.lstsq(M,b,rcond=-1)
endLSnp = end_time(startLSnp)

print(f"Solved (m x n): ({m},{n}) in {endLS} msec, w/ np in {endLSnp} msec \
- Reverting and reconstruction: {endQR} msec, w/ np took: {endQRnp} msec")
print(f"QR error: {np.linalg.norm( M - QR )/np.linalg.norm(QR)} \
- QR error w/ np: {np.linalg.norm( M - QRnp )/np.linalg.norm(QRnp)}\n")
