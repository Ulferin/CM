from abc import ABCMeta, abstractmethod
from sklearn.neural_network import MLPRegressor
import numpy as np


class Optimizer(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self, eta, eps=1e-5):
        self.eta = eta
        self.eps = eps


    @abstractmethod
    def update_mini_batch(self, nn, mini_batch):           
        pass


    # @abstractmethod
    # def get_updates(self, grads):
    #     pass


    @abstractmethod
    def iteration_end(self, nn):      
        pass



class SGD(Optimizer):

    def __init__(self, eta, eps=1e-5):
        super().__init__(eta, eps=eps)


    def update_mini_batch(self, nn, mini_batch):
        """Updates weights and biases of the specified Neural Network object
        :nn: by using the current mini-batch samples :mini_batch:. Uses a
        regularized momentum based approach for the weights update.
        Hyperparameters must be configured directly on the :nn: object.

        Parameters
        ----------
        nn : Object
            Neural Network to use for updates and gradient computation.
        
        mini_batch : np.ndarray
            mini-batch samples to use for updates.
        """
        size = len(mini_batch[0])

        # Nesterov update
        if nn.nesterov:
            nn.weights = [w + nn.momentum * wv
                            for w, wv in zip(nn.weights, nn.wvelocities)]
            nn.biases = [b + nn.momentum * bv
                            for b, bv in zip(nn.biases, nn.bvelocities)]

        # Compute current gradient
        nabla_b, nabla_w = nn._compute_grad(mini_batch)

        # Updates velocities with the current momentum coefficient
        nn.wvelocities = [nn.momentum * velocity - (self.eta/size)*nw
                          for velocity,nw 
                          in zip(nn.wvelocities, nabla_w)]
        nn.bvelocities = [nn.momentum * velocity - (self.eta/size)*nb
                          for velocity,nb
                          in zip(nn.bvelocities, nabla_b)]

        # Updates weights
        nn.weights = [w + velocity - (nn.lmbda/size) * w
                      for w, velocity
                      in zip(nn.weights, nn.wvelocities)]
        nn.biases = [b + velocity
                     for b, velocity
                     in zip(nn.biases, nn.bvelocities)]

    def iteration_end(self, nn):
        """Checks if the optimizer has reached an optimal state.

        Parameters
        ----------
        nn : Object
            Neural Network object being trained using this optimizer.

        Returns
        -------
        bool
            Boolean value indicating whether the optimizer has reached an
            optimal state.
        """        

        if nn.ngrad < self.eps:
            return True

        return False



class Adam(Optimizer):
    def __init__(self, eta, eps=0.00001, beta1=0.9, beta2=0.999):
        super().__init__(eta, eps)
        self.beta1 = beta1
        self.beta2 = beta2
        self.t = 0
        self.offset = 1e-8

        self.first_moment = []
        self.second_moment = []


    def update_mini_batch(self, nn, mini_batch):
        size = len(mini_batch[0])
        self.t += 1

        nabla_b, nabla_w = nn._compute_grad(mini_batch)
        grads = nabla_w + nabla_b

        # Initialize/update gradient accumulation variable
        if len(self.first_moment) == 0:
            self.first_moment = [0]*len(grads)
            self.second_moment = [0]*len(grads)

        self.first_moment = [
            self.beta1 * m + (1-self.beta1)*g
            for m, g in zip(self.first_moment, grads)
        ]

        self.second_moment = [
            self.beta2 * v + (1 - self.beta2)*(g ** 2)
            for v, g in zip(self.second_moment, grads)
        ]

        self.learning_rate = (self.eta
            * np.sqrt(1 - self.beta2**self.t)
            / (1 - self.beta1**self.t))

        params = nn.weights + nn.biases
        updates = [
            -self.learning_rate * fm / (np.sqrt(sm) + self.offset)
            for fm, sm in zip(self.first_moment, self.second_moment)]

        for param, update in zip((p for p in params), updates):
            param += update
            param -= np.sign(param)*nn.lmbda
            # param -= nn.lmbda/size*param


    def iteration_end(self, nn):
        """Checks if the optimizer has reached an optimal state.

        Parameters
        ----------
        nn : Object
            Neural Network object being trained using this optimizer.

        Returns
        -------
        bool
            Boolean value indicating whether the optimizer has reached an
            optimal state.
        """        

        if nn.ngrad < self.eps:
            return True

        return False
    


class SGM(Optimizer):

    def __init__(self, eta, eps=1e-5):
        super().__init__(eta, eps=eps)
        self.step = eta
        self.offset = 1e-8
        self.r_b = []
        self.r_w = []


    def update_mini_batch(self, nn, mini_batch):
        """Updates weights and biases of the specified Neural Network object
        :nn: by using the current mini-batch samples :mini_batch:. Uses a
        diminishing square summable step size as specified by AdaGrad.
        Hyperparameters must be configured directly on the :nn: object.

        Parameters
        ----------
        nn : Object
            Neural Network to use for updates and gradient computation.
        
        mini_batch : np.ndarray
            mini-batch samples to use for updates.
        """       

        # Compute current gradient
        size = len(mini_batch[0])
        nabla_b, nabla_w = nn._compute_grad(mini_batch)
        nabla_b = [nb / size for nb in nabla_b]
        nabla_w = [nw / size for nw in nabla_w]

        # Initialize/update gradient accumulation variable
        if len(self.r_w) == 0:
            self.r_b = [0]*len(nabla_b)
            self.r_w = [0]*len(nabla_w)

        self.r_b = [gb + nb**2 for gb, nb in zip(self.r_b, nabla_b)]
        self.r_w = [gw + nw**2 for gw, nw in zip(self.r_w, nabla_w)]

        # Compute current update
        nabla_b = [nb / (np.sqrt(r_b) + self.offset)
                   for nb, r_b in zip(nabla_b, self.r_b)]
        nabla_w = [nw / (np.sqrt(r_w) + self.offset)
                   for nw, r_w in zip(nabla_w, self.r_w)]

        # Update weights/biases
        nn.weights = [w - self.step*nw - (nn.lmbda/size) * w
                      for w, nw in zip(nn.weights, nabla_w)]
        nn.biases = [b - self.step*nb
                     for b, nb in zip(nn.biases, nabla_b)]


    def iteration_end(self, nn):      
        """Checks if the optimizer has reached an optimal state.
        Updates the current best value for the objective function
        and the current reference value. 

        Parameters
        ----------
        nn : Object
            Neural Network object being trained using this optimizer.

        Returns
        -------
        bool
            Boolean value indicating whether the optimizer has reached an
            optimal state.
        """          

        if nn.ngrad < self.eps:
            return True

        return False