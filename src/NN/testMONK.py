import pandas as pd
from src.NN.NC import NC
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


d_name = 'monks-3'
X_train, X_test, y_train, y_test = load_monk(f"data/MONK/{d_name}")
X_train, X_test = prepare_data(X_train, X_test)

# Loads the input and output layers shape
input_units = X_train.shape[1]
output_units = y_train.reshape(-1,1).shape[1]

# Builds the training data for the NN
training_data = [ (x.reshape(1,-1),y) for x,y in zip(X_train, y_train)]
test_data = [ (x.reshape(1,-1),y) for x,y in zip(X_test, y_test)]

hidden1 = [2, 5]
epochs = [250, 500, 1000, 3000]
batch = [10, 20, 50]
eta = [0.001, 0.2, 1, 3, 7]

for ep in epochs:
    for h1 in hidden1:
        for b in batch:
            for e in eta:
                net = NC([input_units, h1, output_units], 0, debug=False)
                net.SGD(training_data, epochs=ep, batch_size=b, eta=e, test_data=test_data)
                best = net.best_score()
                net.plot_score(f"MONK/{d_name}")
                print(f"Best for (ep: {ep}, h1: {h1}, b: {b}, e: {e}) is:  {best}")


#net = NC([input_units, 5, output_units], 0, debug=False)
#net.SGD(training_data, epochs=3000, batch_size=10, eta=1, test_data=test_data)
#net.plot_score(d_name)
#best = net.best_score()
