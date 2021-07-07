import pandas as pd
from src.NN.NC import NC
from sklearn.preprocessing import OneHotEncoder

import sys

def load_monk(name):
    """ Loads from the provided dataset name both the training set and the test set.
    
    :return: Training and test set examples with the associated dependend variables.
    """
    train = pd.read_csv(f"{name}.train", sep=' ', header=None, index_col=8)
    test = pd.read_csv(f"{name}.test", sep=' ', header=None, index_col=8)
    
    X_train = train.iloc[:,2:].values
    y_train = train.iloc[:,1].values
    X_test = test.iloc[:,2:].values
    y_test = test.iloc[:,1].values
    
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


d_name = 'monks-2'
X_train, X_test, y_train, y_test = load_monk(f"data/{d_name}")
X_train, X_test = prepare_data(X_train, X_test)

# Loads the input and output layers shape
input_units = X_train.shape[1]
output_units = y_train.reshape(-1,1).shape[1]

# Builds the training data for the NN
training_data = X_train.reshape(X_train.shape[0], 1, -1)
test_data = X_test.reshape(X_test.shape[0], 1, -1)

hidden1 = [2, 5]
epochs = [250, 500]
batch = [10, 32]
eta = [0.0001, 0.001, 0.01, 0.1]
lmbda = [0.001, 0.01, 0.1]
momentum = [0.2, 0.5, 0.9]


if __name__ == '__main__':
    test = sys.argv[1]

    h1 = 16
    h2 = 32
    activation = 'Lrelu'
    lmbda = 0
    momentum = 0.5
    epochs = 3000
    batch_size = 32
    eta = 0.001

    if test == 'grid':
        for ep in epochs:
            for h1 in hidden1:
                for b in batch:
                    for e in eta:
                        for l in lmbda:
                            for m in momentum:
                                net = NC([input_units, h1, output_units], 0, 'relu', lmbda=l, momentum=m, debug=False)
                                net.SGD((training_data, y_train), epochs=ep, batch_size=b, eta=e, test_data=(test_data, y_test))
                                print(f"The best score for ep:{ep}, h1:{h1}, b:{b}, e:{e}, l:{l}, m:{m} was: {net.best_score()}")
                                net.plot_score(f"MONK/{d_name}")

    elif test == 'std':
        net = NC([input_units, 5, output_units], 0, activation='relu', lmbda=0.001, momentum=0.5, debug=False)
        net.SGD((training_data, y_train), epochs=1000, batch_size=32, eta=0.01, test_data=(test_data, y_test))
        print(f"The best score for ep:{epochs}, h1:{h1}, h2:{h2}, b:{batch_size}, e:{eta}, l:{lmbda}, m:{momentum} was: {net.best_score()}")
        net.plot_score(f"MONK/{d_name}")

    elif test == 'sub':
        net = NC([input_units, 10, output_units], 0, activation='relu', lmbda=0.001, momentum=0.5, debug=False)
        net.subgrad((training_data, y_train), epochs=1000, start=2, test_data=(test_data, y_test))
        net.plot_score(f"MONK/{d_name}")
