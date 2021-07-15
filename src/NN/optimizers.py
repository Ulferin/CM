from abc import ABCMeta, abstractmethod

import numpy as np



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



class SGM(Optimizer):

    def __init__(self, training_data, epochs, eta, eps=1e-5, batch_size=None, test_data=None):
        super().__init__(training_data, epochs, eta, eps=eps, batch_size=batch_size, test_data=test_data)
        self.step = eta
        self.x_ref = []
        self.f_ref = np.inf


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

        # Compute search direction
        d = self.step/nn.ngrad

        nn.weights = [w - d*nw for w,nw in zip(nn.weights, nabla_w)]
        nn.biases = [b - d*nb for b,nb in zip(nn.biases, nabla_b)]


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

        self.step = self.eta * (1 / e)

        last_f = nn.score

        # found a better value
        if last_f < self.f_ref:
            self.f_ref = last_f
            self.x_ref = (nn.weights.copy(), nn.biases.copy())

        if nn.ngrad < self.eps:
            return True

        return False