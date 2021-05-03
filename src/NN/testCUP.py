import numpy as np
import sys


from NC import NR
import utils


X_train, X_test, y_train, y_test = utils.load_CUP("../../data/ML-CUP20-TR.csv")

train_tup = X_train.copy().reshape(X_train.shape[0], 1, -1)
test_tup = X_test.copy().reshape(X_test.shape[0], 1, -1)

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

net_tup = NR([input_units, 5, output_units], 0, 'relu2', lmbda=0., momentum=0.9, debug=True)
net_tup.SGD_tup((train_tup, y_train), epochs=100, batch_size=10, eta=0.00001, test_data=(test_tup, y_test))
print(net_tup.best_score())

net = NR([input_units, 5, output_units], 0, 'relu2', lmbda=0., momentum=0.9, debug=True)
net.SGD(training_data, epochs=100, batch_size=10, eta=0.00001, test_data=test_data)
print(net.best_score())
# net.plot_score(f"CUP/cup")