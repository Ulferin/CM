import sys
import time
import random
from datetime import datetime as dt

import numpy as np
import matplotlib.pyplot as plt

from src.LS.LS import LS
from src.utils import *


CUP_TEST = 'CUP'
RANDOM_TEST = 'RANDOM'
SCALING_TEST = 'SCALING'

random.seed(42) # Needed for reproducibility

def generate(m, n):
    """Generates a random dataset starting from the given dimensions.

    Parameters
    ----------
    m : int
        Number of rows for the coefficient matrix (i.e. length of the vector b).
    
    n : int
        Number of columns for the coefficient matrix (i.e. length of the vector
        x in (P)) for LS problems.

    Returns
    -------
    M : np.ndarray
        The coefficient matrix M.

    b : np.ndarray
        Dependent variables vector b
    """    
    M = np.array([[random.gauss(0,1) for _ in range(n)]
                    for _ in range(m) ], dtype=np.single)
    b = np.array([random.gauss(0,1) for r in range(m)], dtype=np.single)
    return M, b.reshape(-1,1)


def scaling (starting_m, m, n, step, t) :
    """Tests the QR factorization and the LS solver for different matrices 
    with scaling m, starting from :starting_m: up to :m:.
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
    print(f"m{'':7} QR A3{'':<5s} delta{'':<5s} QR np{'':<5s} delta{'':<5s} "
          f"LS A3{'':<5s} delta{'':<5s} LS np{'':<5s} delta{'':<5s}\n"
          f"-----------------------------------------------------------------"
          f"-----------------------------------")
    
    ls_scaling = (f"\nm{'':7} residual A3{'':<13s} residual np{'':<13s} "
                  f"reconstruct a3{'':<10s} reconstruct np{'':<10s}\n"
                  f"---------------------------------------------------------"
                  f"-------------------------------------------\n")
    
    stats = ""
    
    time_qr_a3 = []
    time_qr_np = []
    time_ls_a3 = []
    time_ls_np = []
    
    prev_qr_a3 = 0
    prev_qr_np = 0
    prev_ls_a3 = 0
    prev_ls_np = 0
    
    ls = LS()
    mrange = range(starting_m,m,step)
    for m in mrange:
        
        A,b = generate(m,n)
        mean_qr_a3 = 0
        mean_qr_np = 0
        mean_ls_a3 = 0
        mean_ls_np = 0
        
        for i in range(t):
            startQR = dt.now()
            R = ls.qr(A)
            Q = ls.revertQ()
            QR = np.matmul(Q, R)
            endQR = end_time(startQR)
            mean_qr_a3 += endQR

            startQRnp = dt.now()
            Qnp, Rnp = np.linalg.qr(A)
            QRnp = np.matmul(Qnp, Rnp)
            endQRnp = end_time(startQRnp)
            mean_qr_np += endQRnp
            
            # Computes time for LS solver 
            startLS = dt.now()
            res = ls.solve(A,b)
            endLS = end_time(startLS)
            mean_ls_a3 += endLS
            
            # Computes time for LS solver using numpy
            startLSnp = dt.now()
            resnp, _, _, _ = np.linalg.lstsq(A,b,rcond=-1)
            endLSnp = end_time(startLSnp)
            mean_ls_np += endLSnp
        
        mean_qr_a3 = (mean_qr_a3 / t)
        mean_qr_np = (mean_qr_np / t)
        mean_ls_a3 = (mean_ls_a3 / t)
        mean_ls_np = (mean_ls_np / t)
        
        delta_qr_a3 = mean_qr_a3 - prev_qr_a3
        delta_qr_np = mean_qr_np - prev_qr_np
        delta_ls_a3 = mean_ls_a3 - prev_ls_a3
        delta_ls_np = mean_ls_np - prev_ls_np
        
        time_qr_a3.append(mean_qr_a3)
        time_qr_np.append(mean_qr_np)
        time_ls_a3.append(mean_ls_a3)
        time_ls_np.append(mean_ls_np)
        
        prev_qr_a3 = mean_qr_a3
        prev_qr_np = mean_qr_np
        prev_ls_a3 = mean_ls_a3
        prev_ls_np = mean_ls_np
        
        
        residual_a3 = np.linalg.norm( b - np.dot(A, res) )/np.linalg.norm(b)
        residual_np = np.linalg.norm( b - np.dot(A, resnp) )/np.linalg.norm(b)
        reconstruct_a3 = np.linalg.norm( A - QR )/np.linalg.norm(A)
        reconstruct_np = np.linalg.norm( A - QRnp )/np.linalg.norm(A)
        
        stats += (f"Solved (m x n): ({m},{n}) in {mean_ls_a3} msec, "
        f"w/ np in {mean_ls_np} msec - "
        f"Reverting and reconstruction: {mean_qr_a3} msec, "
        f"w/ np took: {mean_qr_np} msec\n"
        f"res error: {residual_a3} - np_res error: {residual_np}\n"
        f"QR error: {reconstruct_a3} - QR error w/ np: {reconstruct_np}\n\n")
        
        print(f"{m:<6} || {mean_qr_a3:8.4f} | {delta_qr_a3:8.4f} | "
              f"{mean_qr_np:8.4f} | {delta_qr_np:8.4f} | {mean_ls_a3:8.4f} | "
              f"{delta_ls_a3:8.4f} | {mean_ls_np:8.4f} | {delta_ls_np:8.4f}")
        
        ls_scaling += (f"{m:<6} || {residual_a3:22} | {residual_np:22} | "
                      f"{reconstruct_a3:22} | {reconstruct_np:22}\n")
        
    print(ls_scaling)
    print(stats)
    
    return time_qr_np, time_qr_a3, time_ls_np, time_ls_a3


def generic_test(M, b, test_type):
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

    print(f"---------- {test_type} DATASET ----------\n"
          f"Solved (m x n): ({m},{n}) in {endLS} msec, w/ np in {endLSnp} msec "
          f"- Reverting and reconstruction: {endQR} msec, "
          f"w/ np took: {endQRnp} msec\n"
          f"res error: {np.linalg.norm(b - M@res)/np.linalg.norm(b)} - "
          f"np_res error: {np.linalg.norm(b - M@resnp)/np.linalg.norm(b)}\n"
          f"QR error: {np.linalg.norm(M - QR)/np.linalg.norm(M)} "
          f"- QR error w/ np: {np.linalg.norm(M - QRnp)/np.linalg.norm(M)}\n")
    
    return res, resnp


if __name__ == "__main__":
    """This file provides various test suites, needed to check the accuracy of
    the implemented methods, as well as printing the execution times for the
    given dimension. In the following are listed all the implemented tests and
    what they're supposed to do.

    - CUP dataset test: ran specifying as first execution parameter the 'cup'
                        string. It needs an additional parameter that represents
                        the CUP dataset as a .csv file. It runs the QR
                        factorization on this dataset with both the implemented
                        QR factorization and the off-the-shelf version with
                        numpy. It checks both the accuracy achieved in the
                        factorization and the execution time.

    - Random dataset test: ran specifying as first execution parameter the
                           'random' string. It needs two more parameters to be
                           specified that describes the maximum number of rows
                           and columns for the dataset. It checks the execution
                           time, the accuracy in the LS problem result and in 
                           the QR factorization for each generated dataset.

    - Scaling test: ran specifying as first execution parameter the 'scaling'
                    string. It checks the scalability of the implemented
                    algorithm, running various random datasets with increasing
                    sizes. It shows the execution times for each dataset as well
                    as the difference in computing times from a dataset
                    dimension to the previous one. We expect for this test to
                    show a linear time increasing with the m dimension of the
                    dataset matrix. Finally, it saves an image showing the time
                    taken for each m dimension of the datasets used for testing.

    """

    test = sys.argv[1]      # test type ('cup', 'random' or 'scaling')

    if test == CUP_TEST:
        M, _, b, _ = load_CUP(sys.argv[2], split=0)
        generic_test(M, b, test)
    elif test == RANDOM_TEST:
        m = int(sys.argv[2])    # number of rows
        n = int(sys.argv[3])    # number of cols
        M, b = generate(m, n)
        generic_test(M, b, test)
    elif test == SCALING_TEST:
        starting_m = int(sys.argv[2])
        m = int(sys.argv[3])
        n = int(sys.argv[4])
        step = int(sys.argv[5])
        t = int(sys.argv[6])
        scaling(starting_m, m, n, step, t)
