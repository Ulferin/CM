import QR
import numpy as np
import random

# Generate random matrix for QR decomposition
def generate(m, n):
    return np.array([ [random.gauss(0,1) for r in range(m)] for c in range(n) ])


# task 2: use qr decomp for polynomial regression example
def polyfit(qr, x, y, n):
    return lsqr(qr, x[:, None]**np.arange(n + 1), y.T)
 

def lsqr(qr, a, b):
    u_list, r = qr.qr(a)
    _, n = r.shape
    implicit = qr.implicit_Qb(b)[:n]
    return np.dot( np.linalg.inv(r), implicit )


# a = np.array(((
#     (12, -51,   4),
#     ( 6, 167, -68),
#     (-4,  24, -41),
# )))

m = 1000
n = 1000

qr = QR.QR()
# u_list, R = qr.qr(M)
# print(f"list of u: {u_list};\n\n R: {R}")
 
x = np.array((0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10))
y = np.array((1, 6, 17, 34, 57, 86, 121, 162, 209, 262, 321))

res = polyfit(qr, x, y, 2)

print("\nreverse: ",np.dot(x[:, None]**np.arange(2 + 1), res))
# print('\npolyfit:\n', res)