from LS import LS
import numpy as np
import random
import sys
import time
import matplotlib.pyplot as plt

from datetime import datetime as dt
import time

CUP_TEST = 'CUP'
RANDOM_TEST = 'RANDOM'
QR_SCALING = 'scaling'


def generate(m, n):
    """Generates a random dataset starting from the given dimensions.

    :param m: Number of rows for the coefficient matrix (i.e. length of the vector b).
    :param n: Number of columns for the coefficient matrix (i.e. length of the vector x in (P)).
    :return: The coefficient matrix M and the dependent variables vector b.
    """

    M = np.array([ [random.gauss(0,1) for r in range(n)] for c in range(m) ], dtype=np.single)
    b = np.array([random.gauss(0,1) for r in range(m)], dtype=np.single)
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
                b_el = np.array([float(line_split[-2]), float(line_split[-1])])
                M.append (row)
                b.append (b_el)

    return np.matrix(M), np.matrix(b)


def QR_scaling (starting_m, m, n, step, t) :
    """Tests the QR factorization for different matrices with m in [200, 5000] and n=50.
    Executes each example for a given amount of time and averages the times accordingly. For each result
    prints the size m and the average execution time, together with the time difference from the previous
    result.

    At the end of the process, saves an image showing the evolution of execution times over the increase
    of dimension m. The resulting image is saved in the resource folder as 'QRscaling_n50.png'.
    """

    print(f"n={n}, m={m}, t={t}")
    print("m\ttime\tdelta")
    time_list = []
    time_np = []
    mrange = range(starting_m,m,step)
    prev_a = 0
    prev = 0
    ls = LS()
    for m in mrange:
        A,_ = generate(m,n)
        mean = 0
        mean_np = 0
        for i in range(t):
            startQR = dt.now()
            R = ls.qr(A)
            Q = ls.revertQ()
            QR = np.dot(Q, R)
            endQR = end_time(startQR)
            mean += endQR

            startQRnp = dt.now()
            Qnp, Rnp = np.linalg.qr(A)
            QRnp = np.dot(Qnp, Rnp)
            endQRnp = end_time(startQRnp)
            mean_np += endQRnp
        
        mean = (mean / t) / 1000
        mean_np = (mean_np / t) / 1000
        delta = mean - prev_a
        delta_np = mean_np - prev
        print(m,"\t",mean,"\t", delta, "\t", mean_np, "\t", delta_np)
        time_list.append(mean)
        time_np.append(mean_np)
        prev_a = mean
        prev = mean_np

    plt.plot (mrange, time_list, "bo-", label="mio")
    # plt.plot(mrange, time_np, "r^-", label="np")
    plt.legend()

    plt.xlabel ("m")
    plt.ylabel ("time (sec)")
    plt.title (f"QR factorizzation of a matrix {m}x{n}")

    plt.gca().set_xlim ((min(mrange)-1, max(mrange)+1))

    plt.savefig(f"../results/QRscaling_n{n}m{m}_d{time.time()}.png")
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
    QR = np.dot(Q, R)
    endQR = end_time(startQR)


    # Computes QR factorization using numpy
    startQRnp = dt.now()
    Qnp, Rnp = np.linalg.qr(M)
    QRnp = np.dot(Qnp, Rnp)
    endQRnp = end_time(startQRnp)

    # Computes time for LS solver using numpy
    startLSnp = dt.now()
    resnp, _, _, _ = np.linalg.lstsq(M,b,rcond=-1)
    endLSnp = end_time(startLSnp)

    print(f"---------- {test_type} DATASET ----------")
    print(f"Solved (m x n): ({m},{n}) in {endLS} msec, w/ np in {endLSnp} msec \
    - Reverting and reconstruction: {endQR} msec, w/ np took: {endQRnp} msec")
    print(f"res error: {np.linalg.norm( b - np.dot(M, res) )/np.linalg.norm(b)} \
    - np_res error: {np.linalg.norm( b - np.dot(M, resnp) )/np.linalg.norm(b)}")
    print(f"QR error: {np.linalg.norm( M - QR )/np.linalg.norm(QR)} \
    - QR error w/ np: {np.linalg.norm( M - QRnp )/np.linalg.norm(QRnp)}\n")


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
