from abc import ABCMeta, abstractmethod

import numpy as np
import time


class Optimizer(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self, training_data, epochs, eta, eps=1e-5, batch_size=None, test_data=None):

        # Store auxiliary informations to pretty-print statistics
        self.batch_size = batch_size
        self.eta = eta
        self.eps = eps
        self.training_size = len(training_data[0])
        self.batches = int(self.training_size/batch_size) if batch_size is not None else 1
        self.grad_est_per_epoch = []

        # Reshape vectors to fit needed shape
        self.training_data = (training_data[0], training_data[1].reshape(training_data[1].shape[0], -1))
        self.test_data = test_data


    @abstractmethod
    def update_mini_batch(self, nn, mini_batch):
        """Updates weights and biases of the specified Neural Network object :nn: by
        using the current mini-batch samples :mini_batch:.

        Parameters
        ----------
        nn : Object
            Neural Network to use for updates and gradient computation.
        mini_batch : np.ndarray
            mini-batch samples to use for updates.
        """              
        pass


    @abstractmethod
    def iteration_end(self, e, nn):
        pass



class SGD(Optimizer):

    def __init__(self, training_data, epochs, eta, eps=1e-5, batch_size=None, test_data=None):
        super().__init__(training_data, epochs, eta, eps=eps, batch_size=batch_size, test_data=test_data)


    def update_mini_batch(self, nn, nabla_b, nabla_w, size):
        """Updates weights and biases of the specified Neural Network object :nn: by
        using the current mini-batch samples :mini_batch:. Uses a regularized momentum
        based approach for the weights update. Hyperparameters must be configured directly
        on the :nn: object.

        Parameters
        ----------
        nn : Object
            Neural Network to use for updates and gradient computation.
        mini_batch : np.ndarray
            mini-batch samples to use for updates.
        """

        # Momentum updates
        nn.wvelocities = [nn.momentum * velocity - (self.eta/size)*nw for velocity,nw in zip(nn.wvelocities, nabla_w)]
        nn.bvelocities = [nn.momentum * velocity - (self.eta/size)*nb for velocity,nb in zip(nn.bvelocities, nabla_b)]

        nn.weights = [w + velocity - (nn.lmbda/size) * w for w,velocity in zip(nn.weights, nn.wvelocities)]
        nn.biases = [b + velocity for b,velocity in zip(nn.biases, nn.bvelocities)]


    def iteration_end(self, e, nn):
        """Checks at each iteration if the optimizer has reached an optimal state.

        Parameters
        ----------
        e : int
            Current epoch of training for the given network :nn:.
        nn : Object
            Neural Network object being trained using this optimizer.

        Returns
        -------
        bool
            Whether the optimizer has reached an optimal state.
        """        

        if nn.ngrad < self.eps:
            return True

        return False


# TODO: implementare Polyak step
# TODO: implementare deflected
class SGM(Optimizer):

    def __init__(self, training_data, epochs, eta, eps=1e-5, batch_size=None, test_data=None, deflected=True):
        super().__init__(training_data, epochs, eta, eps=eps, batch_size=batch_size, test_data=test_data)
        self.step = eta
        self.x_ref = []
        self.f_ref = np.inf
        self.deflected = deflected
        self.gamma = 0.95
        self.offset = 1e-8
        self.gms_b = []
        self.gms_w = []


    def update_mini_batch(self, nn, nabla_b, nabla_w, size):
        """Updates weights and biases of the specified Neural Network object :nn: by
        using the current gradient values :nabla_b: for biases and :nabla_w: for weights.
        Uses a diminishing step-size rule for updates.

        Parameters
        ----------
        nn : Object
            Neural Network to use for updates and gradient computation.
        nabla_b : list
            List of gradients for the biases of each layer of the network.
        nabla_w : list
            List of gradients for the weights of each layer of the network.
        size : int
            Not used in SGM. Only used in SGD optimizer.
        """        

        if self.deflected:

            # if len(self.gms_w) == 0:
            #     self.gms_b = [nb**2 for nb in nabla_b]
            #     self.gms_w = [nw**2 for nw in nabla_w]
            # else:
            #     self.gms_b = [self.gamma*gms_b + (1-self.gamma)*nb**2 for nb, gms_b in zip(nabla_b, self.gms_b)]
            #     self.gms_w = [self.gamma*gms_w + (1-self.gamma)*nw**2 for nw, gms_w in zip(nabla_w, self.gms_w)]
            
            if len(self.gms_w) == 0:
                self.gms_b = [0]*len(nabla_b)
                self.gms_w = [0]*len(nabla_w)

            self.gms_b = [gb + nb**2 for gb, nb in zip(self.gms_b, nabla_b)]
            self.gms_w = [gw + nw**2 for gw, nw in zip(self.gms_w, nabla_w)]

            nabla_b = [nb/(np.sqrt(gms_b)+self.offset) for nb, gms_b in zip(nabla_b, self.gms_b)]
            nabla_w = [nw/(np.sqrt(gms_w)+self.offset) for nw, gms_w in zip(nabla_w, self.gms_w)]

        else:
            # Compute search direction
            nabla_b = [nb/nn.ngrad for nb in nabla_b]
            nabla_w = [nw/nn.ngrad for nw in nabla_w]

        nn.weights = [w - self.step*nw - (nn.lmbda/size) * w for w,nw in zip(nn.weights, nabla_w)]
        nn.biases = [b - self.step*nb for b,nb in zip(nn.biases, nabla_b)]


    def iteration_end(self, e, nn):      
        """Checks at each iteration if the optimizer has reached an optimal state,
        updates the learning rate given the current epoch :e: and the reference values
        for the current optimizer based on results coming from the :nn:.

        Parameters
        ----------
        e : int
            Current epoch of training for the given network :nn:.
        nn : Object
            Neural Network object being trained using this optimizer.

        Returns
        -------
        bool
            Whether the optimizer has reached an optimal state.
        """        

        if not self.deflected: self.step = self.eta * (1 / e)

        last_f = nn.score

        # found a better value
        if last_f < self.f_ref:
            self.f_ref = last_f
            self.x_ref = (nn.weights.copy(), nn.biases.copy())

        if nn.ngrad < self.eps:
            return True

        return False