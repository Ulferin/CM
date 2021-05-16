import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_CUP(name):
    df = pd.read_csv(name, header=None, index_col=0, skiprows=7)

    ml_cup = np.delete(np.genfromtxt(name, delimiter=','), obj=0, axis=1)
    M, b = ml_cup[:, :-2], ml_cup[:, -2:]

    X_train, X_test, y_train, y_test = train_test_split(
            M, b, test_size=0.3, random_state=0)

    return X_train, X_test, y_train, y_test