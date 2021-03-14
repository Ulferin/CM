import QR
import numpy as np
import random

# a = np.array(((
#     (12, -51,   4),
#     ( 6, 167, -68),
#     (-4,  24, -41),
# )))
m = 1000
n = 1000
M = np.array([ [random.gauss(0,1) for r in range(m)] for c in range(n) ])

qr = QR.QR()
u_list, R = qr.qr(M)
# print(f"list of u: {u_list};\n\n R: {R}")

