from abc import ABCMeta, abstractmethod
from sklearn.neural_network import MLPRegressor
import numpy as np


class Optimizer(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self, eta, eps=1e-5, lmbda=0.001):
        self.eta = eta
        self.eps = eps
        self.lmbda = lmbda


    @abstractmethod
    def update_mini_batch(self, nn, mini_batch, parameters):           
        pass


    # @abstractmethod
    # def get_updates(self, grads):
    #     pass


    @abstractmethod
    def iteration_end(self, nn):      
        pass



class SGD(Optimizer):

    def __init__(self, eta, eps=1e-5, lmbda=0.01, momentum=0.9, nesterov=True):
        super().__init__(eta, eps=eps, lmbda=lmbda)
        self.nesterov = nesterov
        self.momentum = momentum

        self.v = []

    def update_mini_batch(self, nn, mini_batch, parameters):
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

        if len(self.v) == 0:
            self.v = [np.zeros_like(param) for param in parameters]

        # Nesterov update
        if self.nesterov:
            for param, velocity in zip((p for p in parameters), self.v):
                param += self.momentum * velocity


        # Compute current gradient
        nabla_b, nabla_w = nn._compute_grad(mini_batch)
        grads = nabla_w + nabla_b

        # Updates velocities with the current momentum coefficient
        self.v = [
            self.momentum * velocity - (self.eta/size)*g
            for velocity, g
            in zip(self.v, grads)
        ]

        for param, update in zip((p for p in parameters), self.v):
            param += update
            param -= self.lmbda*param/size
            #FIXME:  notice here we are applying regularization to biases as well
            #       it would be good to avoid this and only consider weights

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
    def __init__(self, eta, eps=0.00001, beta1=0.9, beta2=0.999, lmbda=0.01):
        super().__init__(eta, eps, lmbda)
        self.beta1 = beta1
        self.beta2 = beta2
        self.t = 0
        self.offset = 1e-8

        self.first_moment = []
        self.second_moment = []


    def update_mini_batch(self, nn, mini_batch, parameters):
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

        # params = nn.weights + nn.biases
        updates = [
            -self.learning_rate * fm / (np.sqrt(sm) + self.offset)
            for fm, sm in zip(self.first_moment, self.second_moment)]

        for param, update in zip((p for p in parameters), updates):
            param += update
            param -= np.sign(param)*nn.lmbda/size
            #FIXME: here we are applying regularization to both weights and
            #       biases, but it should be correct to only apply to weights


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