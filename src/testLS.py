from LS import LS
import numpy as np
import random
import sys
import time
import matplotlib.pyplot as plt

from datetime import datetime as dt

CUP_TEST = 'cup'
RANDOM_TEST = 'random'
QR_SCALING = 'scaling'


def generate(m, n):
    """Generates a random dataset starting from the given dimensions.

    :param m: Number of rows for the coefficient matrix (i.e. length of the vector b).
    :param n: Number of columns for the coefficient matrix (i.e. length of the vector x in (P)).
    :return: The coefficient matrix M and the dependent variables vector b.
    """

    M = np.array([ [random.gauss(0,1) for r in range(n)] for c in range(m) ])
    b = np.array([random.gauss(0,1) for r in range(m)])
    return M, b


def end_time(start):
    """Auxiliary function used to compute the ending time (in msec) given the starting time.

    :param start: Starting time in a datetime format.
    :return: Ending time expressed in msec.
    """

    end = (dt.now() - start)
    return end.seconds * 1000 + end.microseconds / 1000


def load_ML_CUP_dataset ( filename ):
    """Given the filename for a dataset, loads the dataset into the coefficient matrix M and the dependent
    variables vector b.

    :param filename: Dataset file path (relative or absolute).
    :return: The coefficient matrix M and the dependend variables vector b.
    """
	
    M = []
    b = []

    with open (filename) as fin:
        for line in fin:
            if not line=="\n" and not line.startswith ("#"): 
                line_split = line.split(",")
                row = [ float(x) for x in line_split[1:-2] ]
                b_el = float(line_split[-1])
                
                M.append (row)
                b.append (b_el)

    return np.array(M), np.array(b)


def QR_scaling () :
    """Tests the QR factorization for different matrices with m in [200, 5000] and n=50.
    Executes each example for a given amount of time and averages the times accordingly. For each result
    prints the size m and the average execution time, together with the time difference from the previous
    result.

    At the end of the process, saves an image showing the evolution of execution times over the increase
    of dimension m. The resulting image is saved in the resource folder as 'QRscaling_n50.png'.
    """

    print("n=50")
    print("m\ttime\tdelta")
    n = 50
    t = 10
    time_list = []
    mrange = range(200,5200,200)
    prev_a = 0
    ls = LS()
    for m in mrange:
        A,_ = generate(m,n)
        mean = 0
        for i in range(t):
            startQR = dt.now()
            R = ls.qr(A)
            Q = ls.revertQ()
            R_complete = np.zeros((m,n))
            R_complete[:n, :n] = R_complete[:n, :n] + R
            QR = np.dot(Q, R_complete)
            endQR = end_time(startQR)
            mean += endQR
        
        mean = (mean / t) / 1000
        delta = mean - prev_a 
        print(m,"\t",mean,"\t", delta)
        time_list.append(mean)
        prev_a = mean

    plt.plot (mrange, time_list, "bo-")

    plt.xlabel ("m")
    plt.ylabel ("time (sec)")
    plt.title ("Qr factorizzation of a matrix mx50")

    plt.gca().set_xlim ((min(mrange)-1, max(mrange)+1))

    plt.savefig("../results/QRscaling_n50_secs.png")
    plt.clf()


def test_random_dataset(m, n):
    # ---------- TEST ON RANDOM DATASET ----------
    ls = LS()
    M, b = generate(m, n)

    # Computes time for LS solver 
    startLS = dt.now()
    res = ls.solve(M,b)
    endLS = end_time(startLS)

    # Computes time for Q and QR reconstruction
    startQR = dt.now()
    R = ls.qr(M)
    Q = ls.revertQ()
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
    resnp, _, _, _ = np.linalg.lstsq(M,b,rcond=-1)
    endLSnp = end_time(startLSnp)

    print(f"---------- RANDOM DATASET ----------")
    print(f"Solved (m x n): ({m},{n}) in {endLS} msec, w/ np in {endLSnp} msec \
    - Reverting and reconstruction: {endQR} msec, w/ np took: {endQRnp} msec")
    print(f"res error: {np.linalg.norm( b - np.dot(M, res) )/np.linalg.norm(b)} \
    - np_res error: {np.linalg.norm( b - np.dot(M, resnp) )/np.linalg.norm(b)}")
    print(f"QR error: {np.linalg.norm( M - QR )/np.linalg.norm(QR)} \
    - QR error w/ np: {np.linalg.norm( M - QRnp )/np.linalg.norm(QRnp)}\n")

    # ---------- TEST ON RANDOM DATASET ----------


def test_cup():
    # ---------- TEST ON CUP DATASET ----------
    ls = LS()
    M, b = load_ML_CUP_dataset("../data/ML-CUP18-TR.csv")

    m = len(M)
    n = len(M[0])

    # Computes time for LS solver 
    startLS = dt.now()
    res = ls.solve(M,b)

    R = ls.qr(M)
    Q = ls.revertQ()
    R_complete = np.zeros((m,n))
    R_complete[:n, :n] = R_complete[:n, :n] + R
    QR = np.dot(Q, R_complete)

    endLS = end_time(startLS)

    # Computes time for LS solver using numpy
    startLSnp = dt.now()
    resnp, _, _, _ = np.linalg.lstsq(M,b,rcond=-1)

    Qnp, Rnp = np.linalg.qr(M, mode="complete")
    QRnp = np.dot(Qnp, Rnp)

    endLSnp = end_time(startLSnp)

    print(f"---------- CUP DATASET ----------")
    print(f"Solved (m x n): ({len(M)},{len(M[0])}) in {endLS} msec, w/ np in {endLSnp} msec")
    print(f"res error: {np.linalg.norm( b - np.dot(M, res) )/np.linalg.norm(b)} \
    - np_res error: {np.linalg.norm( b - np.dot(M, resnp) )/np.linalg.norm(b)}")
    print(f"QR error: {np.linalg.norm( M - QR )/np.linalg.norm(QR)} \
    - QR error w/ np: {np.linalg.norm( M - QRnp )/np.linalg.norm(QRnp)}\n")

    # ---------- TEST ON CUP DATASET ----------


if __name__ == "__main__":
    test = sys.argv[1]      # test type ('cup' or 'random')

    if test == CUP_TEST:
        test_cup()
    elif test == RANDOM_TEST:
        assert len(sys.argv) == 4, "This kind of test requires 'm' and 'n' dimensions to be defined."
        m = int(sys.argv[2])    # number of rows
        n = int(sys.argv[3])    # number of cols
        test_random_dataset(m, n)
    elif test == QR_SCALING:
        QR_scaling()
