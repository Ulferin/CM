import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from datetime import datetime as dt


def load_CUP(name, split=0.3):
    ml_cup = np.delete(np.genfromtxt(name, delimiter=','), obj=0, axis=1)
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
    """ Loads from the provided dataset name both the training set and the test set.
    
    :return: Training and test set examples with the associated dependend variables.
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
    """ Utility function used to one-hot encode the training and test set of MONK datasets.

    :param X_train: training set
    :param X_test: test set
    :return: one-hot encoded training and test set as numpy array
    """

    enc = OneHotEncoder()
    enc.fit(X_train)
    X_train = enc.transform(X_train).toarray()
    X_test = enc.transform(X_test).toarray()

    return X_train, X_test


def crossValToDf(res, scoring='Accuracy'):
        """
        Utility function that transforms the results obtained from the grid-search approach into a DataFrame.

        Params:
            - res: results coming from the grid-search approach.
            - scoring: string representing the scoring function used in the grid-search approach. Defaults to 'Accuracy'.

        Returns:
            - df_sorted: DataFrame with grid-search results.
        """        

        df1 = pd.concat([pd.DataFrame(res["params"]), pd.DataFrame(res["mean_test_score"], columns=[f"Validation {scoring}"])],axis=1)
        df_sorted = df1.sort_values([f"Validation {scoring}"], ascending=False)
        df_sorted = df_sorted.reset_index(drop=True)
        return df_sorted
        

def end_time(start):
    return (dt.now() - start)