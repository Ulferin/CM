from abc import ABCMeta, abstractmethod

import numpy as np


class Optimizer(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self, eta, eps=1e-5):
        """Initialize the optimizer with the specified learning rate :eta:
        and the desire precision :eps:.

        Parameters
        ----------
        eta : float
            Learning rate for SGD optimizer or starting step size for SGM
            optimizer
        
        eps : float, optional
            Desired precision for gradient norm, by default 1e-5
        """        

        self.eta = eta
        self.eps = eps


    @abstractmethod
    def update_mini_batch(self, nn, mini_batch):           
        pass


    @abstractmethod
    def iteration_end(self, nn):
        """Checks if the optimizer has reached an optimal state.

        Parameters
        ----------
        nn : Object
            Neural Network object being trained by this optimizer
        """        
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
            print(nn.ngrad, self.eps)
            return True

        return False


class SGM(Optimizer):

    # TODO: controllare a cosa servono in questo caso x_ref e f_ref
    def __init__(self, eta, eps=1e-5):
        super().__init__(eta, eps=eps)
        self.step = eta
        self.x_ref = []
        self.f_ref = np.inf
        self.gamma = 0.95
        self.offset = 1e-8
        self.gms_b = []
        self.gms_w = []


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

        size = len(mini_batch[0])
        nabla_b, nabla_w = nn._compute_grad(mini_batch)

        if len(self.gms_w) == 0:
            self.gms_b = [0]*len(nabla_b)
            self.gms_w = [0]*len(nabla_w)

        self.gms_b = [gb + nb**2 for gb, nb in zip(self.gms_b, nabla_b)]
        self.gms_w = [gw + nw**2 for gw, nw in zip(self.gms_w, nabla_w)]

        nabla_b = [nb / (np.sqrt(gms_b) + self.offset)
                   for nb, gms_b in zip(nabla_b, self.gms_b)]
        nabla_w = [nw / (np.sqrt(gms_w) + self.offset)
                   for nw, gms_w in zip(nabla_w, self.gms_w)]

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

        last_f = nn.score

        # found a better value
        if last_f < self.f_ref:
            self.f_ref = last_f
            self.x_ref = (nn.weights.copy(), nn.biases.copy())

        if nn.ngrad < self.eps:
            print(f"SGM - grad:{nn.ngrad}, eps: {self.eps}")
            return True

        return False