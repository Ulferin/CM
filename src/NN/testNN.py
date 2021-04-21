import numpy as np
from Network import Network
import utils

# x = np.array([1,2,3,4,5])
X_train, X_test, y_train, y_test = utils.load_CUP("../../data/ML-CUP20-TR.csv")

net = Network([X_train.shape[1], 2, y_train.shape[1]], 0)
out = net.feedforward(X_train[0])
print(out)

