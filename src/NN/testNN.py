import numpy as np
from numpy.core.fromnumeric import ndim
from Network import Network

x = np.array([1,2,3,4,5])

net = Network([5, 2, 1], 0)
out = net.feedforward(x)
print(out)

