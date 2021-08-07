import sys
import time
from datetime import datetime as dt

import numpy as np
import matplotlib.pyplot as plt

from src.LS.LS import LS
from src.utils import *


CUP_TEST = 'CUP'
MONK_TEST = 'MONK'
RANDOM_TEST = 'RANDOM'
QR_SCALING = 'SCALING'

# TODO: Integrare con lo stesso test nel notebook
def QR_scaling (starting_m, m, n, step, t) :    
    """Tests the QR factorization for different matrices with scaling m,
    starting from :starting_m: up to :m:.
    Executes each example for a given amount of time :t: and averages the times
    accordingly. For each result prints the size m and the average execution
    time, together with the time difference from the previous result, expressed
    as a delta. It compares the execution with the off-the-shelf-solver from
    Numpy library.

    At the end of the process, saves an image showing the evolution of execution
    times over the increasing dimension m.

    Parameters
    ----------
    starting_m : int
        Starting value for m dimension
    m : int
        Last value for m dimension
    n : int
        Fixed n dimension
    step : int
        Step size to use when scaling the m dimension
    t : int
        Amount of time to repeat the same experiment, it determines the
        averaging factor too.
    """

    print(f"n={n}, m={m}, t={t}")
    print(f"m{'':10} time{'':<6s} delta{'':<5s} time_np{'':<3s}"
          f"delta_np{'':<7s}")
    print("----------------------------------------------------")
    
    # Execution statistics
    time_list = []
    time_np = []
    prev_a = 0
    prev = 0
    
    ls = LS()
    mrange = range(starting_m,m,step)
    for m in mrange:
        A,_ = generate(m,n)
        mean = 0
        mean_np = 0
        for i in range(t):
            startQR = dt.now()
            R = ls.qr(A)
            Q = ls.revertQ()
            QR = np.matmul(Q, R)
            endQR = end_time(startQR)
            mean += endQR

            startQRnp = dt.now()
            Qnp, Rnp = np.linalg.qr(A)
            QRnp = np.matmul(Qnp, Rnp)
            endQRnp = end_time(startQRnp)
            mean_np += endQRnp
        
        mean = (mean / t)
        mean_np = (mean_np / t)
        delta = mean - prev_a
        delta_np = mean_np - prev

        print(f"{m:<6} || {mean:8.4f} | {delta:8.4f} | {mean_np:8.4f} | {delta_np:8.4f}")

        time_list.append(mean)
        time_np.append(mean_np)
        prev_a = mean
        prev = mean_np

    plt.plot (mrange, time_list, "bo-", label="mio")
    # plt.plot(mrange, time_np, "r^-", label="np")
    plt.legend()

    plt.xlabel ("m")
    plt.ylabel ("time (msec)")
    plt.title (f"QR factorizzation of a matrix {m}x{n}")

    plt.gca().set_xlim ((min(mrange)-1, max(mrange)+1))

    plt.savefig(f"report_tests/LS/Scaling/QRscaling_n{n}m{m}_d{time.time()}.png")
    plt.clf()


def automatized_test(M, b, test_type):
    ls = LS()
    m = M.shape[0]
    n = M.shape[1]

    # Computes time for LS solver 
    startLS = dt.now()
    res = ls.solve(M,b)
    endLS = end_time(startLS)

    # Computes time for Q and QR reconstruction
    startQR = dt.now()
    R = ls.qr(M)
    Q = ls.revertQ()
    QR = np.matmul(Q, R)
    endQR = end_time(startQR)


    # Computes QR factorization using numpy
    startQRnp = dt.now()
    Qnp, Rnp = np.linalg.qr(M)
    QRnp = np.matmul(Qnp, Rnp)
    endQRnp = end_time(startQRnp)

    # Computes time for LS solver using numpy
    startLSnp = dt.now()
    resnp, _, _, _ = np.linalg.lstsq(M,b,rcond=-1)
    endLSnp = end_time(startLSnp)


    print(f"---------- {test_type} DATASET ----------")
    print(f"Solved (m x n): ({m},{n}) in {endLS} msec, w/ np in {endLSnp} msec \
    - Reverting and reconstruction: {endQR} msec, w/ np took: {endQRnp} msec")
    print(f"res error: {np.linalg.norm( b - np.dot(M, res) )/np.linalg.norm(b)} "\
    f"- np_res error: {np.linalg.norm( b - np.dot(M, resnp) )/np.linalg.norm(b)}")
    print(f"QR error: {np.linalg.norm( M - QR )/np.linalg.norm(M)} "\
    f"- QR error w/ np: {np.linalg.norm( M - QRnp )/np.linalg.norm(M)}\n")


if __name__ == "__main__":
    """This file provides various test suites, needed to check the accuracy of the implemented
    methods, as well as printing the execution times for the given dimension. In the following
    are listed all the implemented tests and what they're supposed to do.

    - CUP dataset test: ran specifying as first execution parameter the 'cup' string. It needs
                        an additional parameter that represents the CUP dataset as a .csv file.
                        It runs the QR factorization on this dataset with both the implemented
                        QR factorization and the off-the-shelf version with numpy. It checks both
                        the accuracy achieved in the factorization and the execution time.

    - Random dataset test: ran specifying as first execution parameter the 'random' string. It needs
                           two more parameters to be specified that describes the maximum number of rows
                           and columns for the dataset. It checks the execution time, the accuracy in the
                           LS problem result and in the QR factorization for each generated dataset.

    - Scaling test: ran specifying as first execution parameter the 'scaling' string. It checks the scalability
                    of the implemented algorithm, running various random datasets with increasing sizes.
                    It shows the execution times for each dataset as well as the difference in computing times
                    from a dataset dimension to the previous one. We expect for this test to show a linear time
                    increasing with the m dimension of the dataset matrix.
                    Finally, it saves an image showing the time taken for each m dimension of the datasets used for testing.

    """

    test = sys.argv[1]      # test type ('cup', 'random' or 'scaling')

    if test == CUP_TEST:
        assert len(sys.argv) == 3, "This kind of test requires dataset path to be defined."
        M, b = load_ML_CUP_dataset(sys.argv[2])
        automatized_test(M, b, test)
    elif test == MONK_TEST:
        assert len(sys.argv) == 3, "This kind of test requires dataset path to be defined."
        M, b = load_monk(sys.argv[2])
        automatized_test(M, b, test)
    elif test == RANDOM_TEST:
        assert len(sys.argv) == 4, "This kind of test requires 'm' and 'n' dimensions to be defined."
        m = int(sys.argv[2])    # number of rows
        n = int(sys.argv[3])    # number of cols
        M, b = generate(m, n)
        automatized_test(M, b, test)
    elif test == QR_SCALING:
        assert len(sys.argv) == 7, "This kind of test requires 'starting_m', 'm', 'n', 'step' and 't'."
        starting_m = int(sys.argv[2])
        m = int(sys.argv[3])
        n = int(sys.argv[4])
        step = int(sys.argv[5])
        t = int(sys.argv[6])
        QR_scaling(starting_m, m, n, step, t)
