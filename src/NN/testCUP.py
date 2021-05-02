import numpy as np
import sys


from NC import NR
import utils

# x = np.array([1,2,3,4,5])
X_train, X_test, y_train, y_test = utils.load_CUP("../../data/ML-CUP20-TR.csv")

# Loads the input and output layers shape
input_units = X_train.shape[1]
output_units = y_train.shape[1]

# Builds the training data for the NN
training_data = [ (x.reshape(1,-1),y) for x,y in zip(X_train, y_train)]
test_data = [ (x.reshape(1,-1),y) for x,y in zip(X_test, y_test)]

hidden1 = [10, 20, 50, 100, 200]
hidden2 = [10, 50, 100]
epochs = [100, 250, 1000]
batch = [10, 20, 40, 100]
eta = [0.1, 0.2, 0.5, 1, 2]

# for h1 in hidden1:
#     for h2 in hidden2:
#         for ep in epochs:
#             for b in batch:
#                 for e in eta:
#                     net = Network([input_units, h1, h2, output_units], 0)
#                     net.SGD(training_data, ep, b, e, test_data)
#                     net.best_score()

net = NR([input_units, 5, 5, output_units], 0, 'relu2', lmbda=0.1, momentum=0.5, debug=True)
net.SGD(training_data, epochs=1000, batch_size=10, eta=0.001, test_data=test_data)
print(net.best_score())
net.plot_score(f"CUP/cup")