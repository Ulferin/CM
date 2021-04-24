import pandas as pd
from sklearn.model_selection import train_test_split

def load_CUP(name):
    df = pd.read_csv(name, header=None, index_col=0, skiprows=7)
    M = df.iloc[:,:-2]
    b = df.iloc[:,-2:-1]

    X_train, X_test, y_train, y_test = train_test_split(
            M.values, b.values, test_size=0.1, random_state=0)

    return X_train, X_test, y_train, y_test