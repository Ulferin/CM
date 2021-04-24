import pandas as pd
from Network import Network
from sklearn.preprocessing import OneHotEncoder

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


X_train, X_test, y_train, y_test = load_monk('../../data/monks-1')
X_train, X_test = prepare_data(X_train, X_test)

# Loads the input and output layers shape
input_units = X_train.shape[1]
output_units = y_train.reshape(-1,1).shape[1]

# Builds the training data for the NN
training_data = [ (x.reshape(1,-1),y) for x,y in zip(X_train, y_train)]
test_data = [ (x.reshape(1,-1),y) for x,y in zip(X_test, y_test)]
net = Network([input_units, 5, output_units], 0, debug=False)
net.SGD(training_data, epochs=300, batch_size=10, eta=5, test_data=test_data)
net.best_score()