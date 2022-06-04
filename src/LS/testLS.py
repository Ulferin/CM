import sys
import random

from src.LS.utils import *
from src.utils import *


CUP_TEST = 'CUP'
RANDOM_TEST = 'RANDOM'
SCALING_TEST = 'SCALING'

random.seed(42) # Needed for reproducibility


if __name__ == "__main__":
    """Implements various tests to check the accuracy and performance of the
    implemented methods. The types of implemented tests are:

    - CUP dataset test: runs the QR factorization and LS solver on cup dataset
                        with both the implemented solver and the off-the-shelf
                        version with numpy. It checks both the accuracy achieved
                        in the solution and the execution time.

    - Random dataset test: runs the QR factorization and LS solver on a randomly
                           generated dataset with both the implemented solver and
                           the numpy one. Checks both accuracy and execution time.

    - Scaling test: Checks the scalability of the implemented algorithm by
                    running the same test over random datasets with increasing
                    sizes. Shows both execution times and accuracy in the LS
                    solution.

    """

    if len(sys.argv) < 2:
        print("You must provide a test type!")
        exit()

    test = sys.argv[1]      # test type ('CUP', 'RANDOM' or 'SCALING')

    if test == CUP_TEST:
        if len(sys.argv) != 3:
            print("Usage: python -m src.LS.testLS 'CUP' dataset_full_path")
            exit()

        M, _, b, _ = load_CUP(sys.argv[2], split=0)
        res, resnp = generic_test(M, b, test)
        analizeCond(M, b, res, resnp)

    elif test == RANDOM_TEST:
        if len(sys.argv) != 4:
            print("Usage: python -m src.LS.testLS 'RANDOM' m n")
            exit()

        m = int(sys.argv[2])    # number of rows
        n = int(sys.argv[3])    # number of cols
        M, _ = generate(m, n)
        sol = np.random.rand(n)
        b = M@sol

        res, resnp = generic_test(M, b, test)
        analizeCond(M, b, res, resnp, sol)
        
    elif test == SCALING_TEST:
        if len(sys.argv) != 7:
            print("Usage: python -m src.LS.testLS 'SCALING' start_m m_last n step_m repeat_t")
            exit()
        
        starting_m = int(sys.argv[2])
        last_m = int(sys.argv[3])
        n = int(sys.argv[4])
        step = int(sys.argv[5])
        repeat = int(sys.argv[6])
        time_qr_np, time_qr_a3, time_ls_np, time_ls_a3 = scaling(starting_m, last_m, n, step, repeat)
        plot_stats(time_qr_np, time_qr_a3, time_ls_np, time_ls_a3, range(starting_m, last_m+step, step), n, save=True)

 