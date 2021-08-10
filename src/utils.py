import pandas as pd

import numpy as np
import random

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from datetime import datetime as dt


def load_CUP(file_path, split=0.3):
    """Loads the CUP dataset from :file_path: with a default
    splitting strategy of 30% for the internal test set.

    Parameters
    ----------
    file_path : string
        Location of the dataset file

    split : float, optional
        Amount of data to hold out for the internal
        test set, by default 0.3

    Returns
    -------
    X_train: np.ndarray
        Training samples

    X_test: np.ndarray
        Test samples

    y_train: np.ndarray
        Training samples' ground truth

    y_test: np.ndarray
        Test samples' ground truth
    """    
    ml_cup = np.delete(np.genfromtxt(file_path, delimiter=','), obj=0, axis=1)
    M, b = ml_cup[:, :-2], ml_cup[:, -2:]

    if split > 0:
        X_train, X_test, y_train, y_test = train_test_split(
                M, b, test_size=split, random_state=0)
    else:
        X_train = M
        y_train = b
        X_test = None
        y_test= None

    return X_train, X_test, y_train, y_test


def load_monk(name):
    """Loads from the provided :name: the associated training and test set.

    Parameters
    ----------
    name : string
        Name of the monk dataset to load.
        Inputs can be:
            · monk1
            · monk2
            · monk3

    Returns
    -------
    X_train: np.ndarray
        Training samples

    X_test: np.ndarray
        Test samples

    y_train: np.ndarray
        Training samples' ground truth

    y_test: np.ndarray
        Test samples' ground truth
    """
    train = pd.read_csv(f"{name}.train", sep=' ', header=None, index_col=8)
    test = pd.read_csv(f"{name}.test", sep=' ', header=None, index_col=8)
    
    X_train = train.iloc[:,2:].values
    y_train = (train.iloc[:,1].values).reshape(-1,1)
    X_test = test.iloc[:,2:].values
    y_test = (test.iloc[:,1].values).reshape(-1,1)
    
    X_train, X_test = prepare_data(X_train, X_test)

    return X_train, X_test, y_train, y_test


def prepare_data(X_train, X_test):
    """Utility function used to one-hot-encode training and test set
    of MONK datasets.

    Parameters
    ----------
    X_train : np.ndarray
        Training data samples to one-hot encode

    X_test : np.ndarray
        Test data samples to one-hot encode

    Returns
    -------
    X_train: np.ndarray
        One-hot encoded training samples

    X_test: np.ndarray
        One-hot encoded test samples
    """

    enc = OneHotEncoder()
    enc.fit(X_train)
    X_train = enc.transform(X_train).toarray()
    X_test = enc.transform(X_test).toarray()

    return X_train, X_test


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

    M = np.array([ [random.gauss(0,1) for _ in range(n)]
                    for _ in range(m) ], dtype=np.single)
    b = np.array([random.gauss(0,1) for r in range(m)], dtype=np.single)
    return M, b.reshape(-1,1)


def crossValToDf(res, scoring='Accuracy'):
    """Transofrms the results obtained via grid-search into a pandas DataFrame.

    Parameters
    ----------
    res : dictionary
        Results of a grid-search execution
    scoring : string, optional
        String representing the scoring function used in the grid-search,
        by default 'Accuracy'

    Returns
    -------
    DataFrame
        DataFrame with grid-search results
    """ 

    df1 = pd.concat(
            [pd.DataFrame(res["params"]), pd.DataFrame(res["mean_test_score"],
            columns=[f"Validation {scoring}"])],
            axis=1)
    df_sorted = df1.sort_values([f"Validation {scoring}"], ascending=False)
    df_sorted = df_sorted.reset_index(drop=True)
    return df_sorted
        

def end_time(start):
    """Computes elapsed time since the :start: datetime. Returns the time
    expressed in milliseconds.

    Parameters
    ----------
    start : DateTime
        Starting time.

    Returns
    -------
    DateTime
        Elasped time since :start: in milliseconds.
    """    
    end = (dt.now() - start)
    return end.seconds*1000 + end.microseconds/1000