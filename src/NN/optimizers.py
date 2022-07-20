import numpy as np
from abc import ABCMeta, abstractmethod


class Optimizer(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self, learning_rate_init, tol=1e-5):
        self.learning_rate_init = learning_rate_init
        self.tol = tol

        self.updates_norm = []
        self.layer_updates_norm = []


    @abstractmethod
    def update_parameters(self, parameters, grads):
        pass


    def iteration_end(self, ngrad):
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

        if ngrad < self.tol:
            return True

        return False



class sgd(Optimizer):

    def __init__(self, learning_rate_init, tol=1e-5, momentum=0.9,
            nesterovs_momentum=True):
        super().__init__(learning_rate_init, tol=tol)
        self.nesterovs_momentum = nesterovs_momentum
        self.momentum = momentum

        self.v = []

    def update_parameters(self, parameters, grads):
        # TODO: maybe we can add definition of the optimizer and description of
        #       its properties. Like a summary of the things we have written in
        #       the report
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

        if len(self.v) == 0:
            self.v = [np.zeros_like(param) for param in parameters]

        # Updates velocities with the current momentum coefficient
        self.v = [
            self.momentum * velocity - self.learning_rate_init*g
            for velocity, g
            in zip(self.v, grads)
        ]
        self.layer_updates_norm = [[] for _ in range(len(parameters))] if self.layer_updates_norm == [] else self.layer_updates_norm
        self.updates_norm.append(np.linalg.norm(np.hstack([u.ravel() for u in self.v])))

        for i, update in enumerate(self.v):
            self.layer_updates_norm[i].append(np.linalg.norm(update.ravel()))


        for param, update in zip((p for p in parameters), self.v):
            param += update
        
        # nesterovs_momentum update
        if self.nesterovs_momentum:
            for param, velocity in zip((p for p in parameters), self.v):
                param += self.momentum * velocity



class Adam(Optimizer):
    # TODO: add method description
    def __init__(self, learning_rate_init, tol=0.00001, beta_1=0.9,
            beta_2=0.999):
        super().__init__(learning_rate_init, tol)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.t = 0
        self.offset = 1e-8

        self.first_moment = []
        self.second_moment = []



    def update_parameters(self, parameters, grads):
        self.t += 1

        # Initialize/update gradient accumulation variable
        if len(self.first_moment) == 0:
            self.first_moment = [0]*len(grads)
            self.second_moment = [0]*len(grads)

        self.first_moment = [
            self.beta_1 * m + (1-self.beta_1)*g
            for m, g in zip(self.first_moment, grads)
        ]

        self.second_moment = [
            self.beta_2 * v + (1 - self.beta_2)*(g ** 2)
            for v, g in zip(self.second_moment, grads)
        ]

        self.learning_rate = (self.learning_rate_init
            * np.sqrt(1 - self.beta_2**self.t)
            / (1 - self.beta_1**self.t))

        updates = [
            -self.learning_rate * fm / (np.sqrt(sm) + self.offset)
            for fm, sm in zip(self.first_moment, self.second_moment)]

        self.layer_updates_norm = [[] for _ in range(len(parameters))] if self.layer_updates_norm == [] else self.layer_updates_norm
        for i, update in enumerate(updates):
            self.layer_updates_norm[i].append(np.linalg.norm(update.ravel()))

        self.updates_norm.append(np.linalg.norm(np.hstack([u.ravel() for u in updates])))

        for param, update in zip((p for p in parameters), updates):
            param += update
