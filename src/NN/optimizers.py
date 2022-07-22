import numpy as np
from abc import ABCMeta, abstractmethod


class Optimizer(metaclass=ABCMeta):
    """Abstract base class for the optimizers used in the implemented neural
    network. Provides the interface for the derived classes implementing
    specific optimization algorithms. Each derived class must implement the
    function :update_parameters: to update the provided parameters (weights and
    biases) according to the provided gradients.
    
    In our case, two derived classes are implemented:
        · SGD: implements the stochastic gradient descent algorithm, with two
               versions, which are classical momentum (A1/cm) and nesterov
               momentum (A1/nag), as specified in the final report.
        · Adam: implements the Adam optimization algorithm (A2).

    Both classes uses the same stopping criterion based on a tolerance parameter
    which can be configured when creating the optimizer object.
    """

    @abstractmethod
    def __init__(self, learning_rate_init, tol=1e-5):
        """Abstract constructor, defines and initializes common hyperparameters
        for all the implemented optimizers, in particular it allows to specify
        the initial learning rate (which is fixed for the whole training process
        in case of the SGD optimizer) and the tolerance on the gradient norm
        to stop the learning process.

        NOTE: the naming convention for the hyperparameters names is dictated by
        the necessity to make the implemented classes compatible with the model
        selection utilities provided by the sklearn framework.

        Parameters
        ----------
        learning_rate_init : float
            Learning rate hyperparameter for the optimizer. Defined as \eta in
            the final report.
        tol : float, optional
            Tolerance hyperparameter used as stopping condition on the gradient
            norm, by default 1e-5
        """        
        self.learning_rate_init = learning_rate_init
        self.tol = tol


    @abstractmethod
    def update_parameters(self, parameters, grads):
        """Updates the provided :parameters: according to the implemented
        optimization strategy. Specific description of the procedure followed
        to update the netowrk parameters based on the values of the gradients
        :grads: is delayed to the derived classes.

        Parameters
        ----------
        parameters : np.ndarray
            Network parameters, namely the weights and the biases, concatenated
            in a single vector.
        grads : np.ndarray
            Gradient of the objective function with respect to the network
            parameters for the current iteration.
        """        
        pass


    def iteration_end(self, ngrad):
        """Implements the termination condition based on a tolerance over the
        gradient norm :ngrad: Checks if the optimizer has reached an optimal
        state.

        Parameters
        ----------
        ngrad : float
            Gradient norm of the current neural network parameters.

        Returns
        -------
          : bool
            Boolean value indicating whether the optimizer has reached an
            optimal state. Return True whenever the norm of the gradient is below
            the tolerance :tol: specified upon creation of this object.
        """        

        if ngrad < self.tol:
            return True

        return False



class sgd(Optimizer):

    def __init__(self, learning_rate_init, tol=1e-5, momentum=0.9,
            nesterovs_momentum=True):
        """Defines and initializes SGD specific hyperparameters.
        In particular it allows to specify the fixed learning rate to use during
        the whole process of learning, the tolerance on the gradient norm and the
        momentum specific hyperparameters, like momentum coefficient and nesterov
        momentum.

        Parameters
        ----------
        learning_rate_init : float
            Learning rate hyperparameter for the optimizer. Defined as \eta in
            the final report.
        tol : float, optional
            Tolerance hyperparameter used as stopping condition on the gradient
            norm, by default 1e-5
        momentum : float, optional
            Momentum coefficient, defined as \mu in the report, by default 0.9
        nesterovs_momentum : bool, optional
            Boolean value indicating whether to use nesterov momentum, by default
            True
        """  
        super().__init__(learning_rate_init, tol=tol)
        self.nesterovs_momentum = nesterovs_momentum
        self.momentum = momentum

        self.v = []

    def update_parameters(self, parameters, grads):
        """Updates the provided :parameters: according to the SGD
        optimization strategy. It applies the momentum update rule to the
        :parameters: vector by using the provided :grads: vector. Based on the
        initial configuration of :self.nesterovs_momentum:, the update is either
        classical momentum or nesterov momentum.

        Parameters
        ----------
        parameters : np.ndarray
            Parameters vector over which the update is applied. Updated in place.
        grads : np.ndarray
            Gradient of the objective function with respect to the network
            parameters for the current iteration.
        """ 

        if len(self.v) == 0:
            self.v = [np.zeros_like(param) for param in parameters]

        self.vold = self.v

        # Updates velocities with the current momentum coefficient
        self.v = [
            self.momentum * velocity - self.learning_rate_init*g
            for velocity, g
            in zip(self.vold, grads)
        ]

        for param, update in zip((p for p in parameters), self.v):
            param += update

        
        # nesterovs_momentum update
        if self.nesterovs_momentum:
            for param, velocity, vold in zip((p for p in parameters), self.v, self.vold):
                param += self.momentum * (velocity - vold)



class Adam(Optimizer):
    def __init__(self, learning_rate_init, tol=0.00001, beta_1=0.9,
            beta_2=0.999):
        """Defines and initializes Adam specific hyperparameters.
        In particular it allows to specify the base learning rate, the tolerance
        on the gradient norm, and the beta_1/beta_2 coefficients for first/second
        moment vectors.

        Parameters
        ----------
        learning_rate_init : float
            Base learning rate used to computed the 'per-parameter' learning rate
            for the Adam optimizer.
        tol : float, optional
            Tolerance used as stopping condition on the gradient norm, by
            default 0.00001
        beta_1 : float, optional
            First moment decay rate coefficient, by default 0.9
        beta_2 : float, optional
            Second moment decay rate coefficient, by default 0.999
        """
        super().__init__(learning_rate_init, tol)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.t = 0
        self.offset = 1e-8

        self.first_moment = []
        self.second_moment = []



    def update_parameters(self, parameters, grads):
        """Updates the provided :parameters: according to the Adam optimization
        algorithm. Applies a 'per-parameter' update rule using parameter specific
        learning rates which are computed by retaining the first and second
        moments of the gradients.

        Parameters
        ----------
        parameters : np.ndarray
            Parameters vector over which the update is applied. Updated in place.
        grads : np.ndarray
            Gradient of the objective function with respect to the network
            parameters for the current iteration.
        """

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

        for param, update in zip((p for p in parameters), updates):
            param += update
