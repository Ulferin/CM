import numpy as np
import random
from src.LS.LS import LS
from datetime import datetime as dt
import matplotlib.pyplot as plt

from src.utils import *

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
                    for _ in range(m) ], dtype=float)
    b = np.array([random.gauss(0,1) for r in range(m)], dtype=float)
    return M, b.reshape(-1,1)


def generate_squareK(n, k):
    n = n
    cond_P = k     # Condition number
    log_cond_P = np.log(cond_P)
    exp_vec = np.arange(-log_cond_P/4., log_cond_P * (n)/(4 * (n - 1)), log_cond_P/(2.*(n-1)))
    s = np.exp(exp_vec)
    S = np.diag(s)
    U, _ = np.linalg.qr((np.random.rand(n, n) - 5.) * 200)
    V, _ = np.linalg.qr((np.random.rand(n, n) - 5.) * 200)
    P = U.dot(S).dot(V.T)
    P = P.dot(P.T)

    return P


def analizeCond(M, b, res, resnp, sol=None):
    kappa = np.linalg.cond(M)
    eta = None
    if sol is not None:
        eta = np.linalg.norm(M,2) * np.linalg.norm(sol,2) / np.linalg.norm(M@res)
    theta = np.arcsin(np.linalg.norm(b-M@res, 2)/np.linalg.norm(b,2))
    
    cond_yWRTA = kappa / np.cos(theta)
    cond_yWRTb = 1 / np.cos(theta)
    cond_xWRTA = None
    cond_xWRTb = None
    if sol is not None:
        cond_xWRTb = kappa / (eta*np.cos(theta))
        cond_xWRTA = kappa + (kappa**2 * np.tan(theta))/eta

    print(
        f"{'Theta:':<18} {theta}\n"
        f"{'eta:':<18} {eta}\n"
        f"{'kappa:':<18} {kappa}\n"
        f"{'cos(Theta):':<18} {np.cos(theta)}\n"
        f"{'cond y w.r.t b:':<18} {cond_yWRTb}\n"
        f"{'cond y w.r.t A:':<18} {cond_yWRTA}\n"
        f"{'cond x w.r.t. b:':<18} {cond_xWRTb}\n"
        f"{'cond x w.r.t. A:':<18} {cond_xWRTA}\n")
    
    if sol is not None:
        print(f"{'|x - ~x|/|x|:':<23} {np.linalg.norm(res - sol, 2) / np.linalg.norm(sol, 2)}")
    print(f"{'|res - resnp|/|resnp|:':<23} {np.linalg.norm(res - resnp,2) / np.linalg.norm(resnp,2)}")


def scaling (starting_m, m, n, step, t, linear=True) :
    """Tests the QR factorization and the LS solver for different matrices 
    with scaling m, starting from :starting_m: up to :m:.
    Executes each example for a given amount of time :t: and averages the times
    accordingly. For each result prints the size m and the average execution
    time, together with the time difference from the previous result, expressed
    as a delta. It compares the execution with the off-the-shelf-solver from
    Numpy library.

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


    Returns
    -------
    time_qr_np : list
        List of time statistics for numpy QR factorization
    
    time_qr_a3 : list
        List of time statistics for implemented QR factorization

    time_ls_np : list
        List of time statistics for numpy LS solver

    time_ls_a3 : list
        List of time statistics for implemented LS solver
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
    
    # Time statistics
    time_qr_a3 = []
    time_qr_np = []
    time_ls_a3 = []
    time_ls_np = []
    
    # Needed for delta computation
    prev_qr_a3 = 0
    prev_qr_np = 0
    prev_ls_a3 = 0
    prev_ls_np = 0
    
    ls = LS()
    mrange = range(starting_m,m+step,step)
    for m in mrange:

        A,b = generate(m,n)
        mean_qr_a3 = 0
        mean_qr_np = 0
        mean_ls_a3 = 0
        mean_ls_np = 0

        sol = None
        if linear:
            sol = np.random.rand(n)
            b = A@sol

            res, resnp = generic_test(A, b, f'SCALING - {m}x{n}')
            analizeCond(A, b, res, resnp, sol)

        for i in range(t):
            # Computes time for QR factorization
            startQR = dt.now()
            R = ls.qr(A)
            Q = ls.revertQ()
            QR = np.matmul(Q, R)
            endQR = end_time(startQR)
            mean_qr_a3 += endQR

            # Computes time for QR factorization using numpy
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

        # Computing mean time
        mean_qr_a3 = (mean_qr_a3 / t)
        mean_qr_np = (mean_qr_np / t)
        mean_ls_a3 = (mean_ls_a3 / t)
        mean_ls_np = (mean_ls_np / t)
        
        time_qr_a3.append(mean_qr_a3)
        time_qr_np.append(mean_qr_np)
        time_ls_a3.append(mean_ls_a3)
        time_ls_np.append(mean_ls_np)

        # Computing delta
        delta_qr_a3 = mean_qr_a3 - prev_qr_a3
        delta_qr_np = mean_qr_np - prev_qr_np
        delta_ls_a3 = mean_ls_a3 - prev_ls_a3
        delta_ls_np = mean_ls_np - prev_ls_np

        # Saving previous mean time
        prev_qr_a3 = mean_qr_a3
        prev_qr_np = mean_qr_np
        prev_ls_a3 = mean_ls_a3
        prev_ls_np = mean_ls_np
        
        # Computing accuracies        
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
    """Perform a test for both QR factorization algorithm and LS solver over the
    given data matrix :M: and dependent variables :b:.

    Parameters
    ----------
    M : np.ndarray
        Data matrix.

    b : np.ndarray
        Dependend variables vector.

    test_type : string
        Name of the performed test.

    Returns
    -------
    res : np.ndarray
        Solution vector for the implemented LS solver.

    resnp : np.ndarray
        Solution vecor for the numpy LS solver.
    """    

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
          f"res error: {np.linalg.norm(b - M@res, 2)/np.linalg.norm(b, 2)} - "
          f"np_res error: {np.linalg.norm(b - M@resnp, 2)/np.linalg.norm(b, 2)}\n"
          f"QR error: {np.linalg.norm(M - QR, 2)/np.linalg.norm(M, 2)} "
          f"- QR error w/ np: {np.linalg.norm(M - QRnp, 2)/np.linalg.norm(M, 2)}\n"
          f"Relative error on R: {np.linalg.norm(R - Rnp, 2) / np.linalg.norm(Rnp, 2)}\n"
          f"Relative error on Q: {np.linalg.norm(Q - Qnp, 2) / np.linalg.norm(Qnp, 2)}\n"
          f"Relative error on QR: {np.linalg.norm(QR - QRnp, 2) / np.linalg.norm(QRnp, 2)}\n"
          )
    
    return res, resnp


def analizeCond(M, b, res, resnp, sol=None):
    kappa = np.linalg.cond(M)
    eta = None
    if sol is not None:
        eta = np.linalg.norm(M,2) * np.linalg.norm(sol,2) / np.linalg.norm(M@res)
    theta = np.arcsin(np.linalg.norm(b-M@res, 2)/np.linalg.norm(b,2))

    cond_yWRTA = kappa / np.cos(theta)
    cond_yWRTb = 1 / np.cos(theta)
    cond_xWRTA = None
    cond_xWRTb = None
    if sol is not None:
        cond_xWRTb = kappa / (eta*np.cos(theta))
        cond_xWRTA = kappa + (kappa**2 * np.tan(theta))/eta

    print(
        f"{'Theta:':<18} {theta}\n"
        f"{'eta:':<18} {eta}\n"
        f"{'kappa:':<18} {kappa}\n"
        f"{'cos(Theta):':<18} {np.cos(theta)}\n"
        f"{'cond y w.r.t b:':<18} {cond_yWRTb}\n"
        f"{'cond y w.r.t A:':<18} {cond_yWRTA}\n"
        f"{'cond x w.r.t. b:':<18} {cond_xWRTb}\n"
        f"{'cond x w.r.t. A:':<18} {cond_xWRTA}\n")

    if sol is not None:
        print(f"{'|x - ~x|/|x|:':<23} {np.linalg.norm(res - sol, 2) / np.linalg.norm(sol, 2)}")
    print(f"{'|res - resnp|/|resnp|:':<23} {np.linalg.norm(res - resnp,2) / np.linalg.norm(resnp,2)}\n")


def plot_stats(time_qr_np, time_qr_a3, time_ls_np, time_ls_a3, mrange, n, save=False):
    
    m = mrange.stop - mrange.step
    
    # --- Plotting QR stats ---
    plt.plot(mrange, time_qr_np, "^-", label="(NP)")
    plt.xlabel ("m")
    plt.ylabel ("time (msec)")
    plt.title (f"QR factorization of a matrix {m}x{n} (NP)")
    plt.gca().set_xlim ((min(mrange)-1, max(mrange)+1))

    if save: plt.savefig(f"report_tests/LS/Scaling/QRscaling_np_n{n}m{m}.png")
    else: plt.show()
    plt.clf()

    plt.plot (mrange, time_qr_a3, "o-", label="mio")
    plt.xlabel ("m")
    plt.ylabel ("time (msec)")
    plt.title (f"QR factorization of a matrix {m}x{n} (A3)")
    plt.gca().set_xlim ((min(mrange)-1, max(mrange)+1))

    if save: plt.savefig(f"report_tests/LS/Scaling/QRscaling_a3_n{n}m{m}.png")
    else: plt.show()
    plt.clf()
    
    plt.plot(mrange, time_qr_np, "^-", label="(NP)")
    plt.plot (mrange, time_qr_a3, "o-", label="(A3)")
    plt.legend()
    plt.xlabel ("m")
    plt.ylabel ("time (msec)")
    plt.title (f"QR factorization of a matrix {m}x{n}")
    plt.gca().set_xlim ((min(mrange)-1, max(mrange)+1))

    if save: plt.savefig(f"report_tests/LS/Scaling/QRscaling_comparison_n{n}m{m}.png")
    else: plt.show()
    plt.clf()
    
    
    # --- Plotting LS stats ---
    plt.plot(mrange, time_ls_np, "^-")
    plt.xlabel ("m")
    plt.ylabel ("time (msec)")
    plt.title (f"LS of a matrix {m}x{n} (NP)")
    plt.gca().set_xlim ((min(mrange)-1, max(mrange)+1))

    if save: plt.savefig(f"report_tests/LS/Scaling/LSscaling_np_n{n}m{m}.png")
    else: plt.show()
    plt.clf()

    plt.plot (mrange, time_ls_a3, "o-")
    plt.xlabel ("m")
    plt.ylabel ("time (msec)")
    plt.title (f"LS of a matrix {m}x{n} (A3)")
    plt.gca().set_xlim ((min(mrange)-1, max(mrange)+1))

    if save: plt.savefig(f"report_tests/LS/Scaling/LSscaling_a3_n{n}m{m}.png")
    else: plt.show()
    plt.clf()
    
    plt.plot(mrange, time_ls_np, "^-", label="(NP)")
    plt.plot (mrange, time_ls_a3, "o-", label="(A3)")
    plt.legend()
    plt.xlabel ("m")
    plt.ylabel ("time (msec)")
    plt.title (f"LS of a matrix {m}x{n}")
    plt.gca().set_xlim ((min(mrange)-1, max(mrange)+1))

    if save: plt.savefig(f"report_tests/LS/Scaling/LSscaling_comparison_n{n}m{m}.png")
    else: plt.show()
    plt.clf()