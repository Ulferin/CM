import numpy as np
from Network import Network
import utils

# x = np.array([1,2,3,4,5])
X_train, X_test, y_train, y_test = utils.load_CUP("../../data/ML-CUP20-TR.csv")

# Loads the input and output layers shape
input_units = X_train.shape[1]
output_units = y_train.shape[1]

# Builds the training data for the NN
training_data = [ (x,y) for x,y in zip(X_train, y_train)]
test_data = [ (x,y) for x,y in zip(X_test, y_test)]

net = Network([input_units, 100, 30, output_units], 0)
out = net.feedforward(X_train[0])
net.SGD(training_data, 100, 50, 1, test_data)
print(out)

