# This class represents the Network used in the CM project and it is
# implemented from scratch following the advices taken during the course
# of ML

from abc import ABCMeta, abstractmethod

import numpy as np
from numpy.linalg import norm
from numpy.random import default_rng

from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from sklearn.utils import shuffle

from matplotlib import pyplot as plt

from src.NN.ActivationFunctions import ReLU, Sigmoid, LeakyReLU, Linear
from src.NN.LossFunctions import MeanSquaredError, AccuracyScore
from src.NN.optimizers import SGD, SGM


ACTIVATIONS = {
    'relu': ReLU,
    'Lrelu':LeakyReLU,
    'sigmoid': Sigmoid
}

OPTIMIZERS = {
    'SGD': SGD,
    'SGM': SGM
}


class Network(metaclass=ABCMeta):
    """Abstract class representing a standard Neural Network, also known as Multilayer Perceptron.
    It allows to build a network for both classification and regression tasks by using the
    preferred optimization technique between sub-gradient method and stochastic gradient descent.
    Must be extended to specify which kind of task the network should solve.
    """    
    
    @abstractmethod
    def __init__(self, sizes, seed, activation='sigmoid', lmbda=0.0, momentum=0.0, debug=True):
        """Initializes the network topology based on the given :sizes: which represents the amount
        of units to use for each of the layers of the current network. Builds the weights and bias
        vectors for each layer of the network accordingly. Each layer will be initialized randomly
        following the normal distribution. Various hyperparameters can be specified, like momentum
        and regularization coefficients.
        # TODO: usiamo veramente normal distribution? 

        Parameters
        ----------
        sizes : tuple
            Tuple (i, l1, l2, ..., ln, o) containig the number of units for each layer,
            where the first and last elements represents, respectively, the input layer
            and the output layer units.
        seed : int
            seed for random number generator used for initializing this network weights
            and biases. Needed for reproducibility.
        activation : function, optional
            specifies which activation function to use for the hidden layers of the network,
            by default 'sigmoid'
        lmbda : int, optional
            l2 regularization coefficient, by default 0.0
        momentum : int, optional
            momentum coefficient, by default 0.0
        debug : bool, optional
            debugging flag, by default True
        """        
        
        rng = default_rng(seed)     # needed for reproducibility
        self.training_size = None
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.act = ACTIVATIONS[activation]
        self.momentum = momentum
        self.lmbda = lmbda
        self.last_act = None            # Must be defined by subclassing the Network
        self.g = None
        self.grad_est = None
        self.grad_est_per_epoch = []


        self.biases = [np.zeros_like(l) for l in sizes[1:]]
        self.weights = [rng.normal(0, 1, (y,x))/np.sqrt(x) for x, y in zip(sizes[:-1], sizes[1:])]
        self.wvelocities = [np.zeros_like(weight) for weight in self.weights]
        self.bvelocities = [np.zeros_like(bias) for bias in self.biases]
        self.val_scores = []
        self.train_scores = []

        self.debug = debug


    def _feedforward_batch(self, inp):
        """Performs a feedforward pass through the network to compute the
        output for the current input to the network.

        Parameters
        ----------
        inp : np.ndarray
            Input to the network used to perform the feedforward pass.

        Returns
        -------
        np.ndarray, list, list
            An np.ndarray representing the output related the the current input
            :inp: and two lists representing respectively the list of inputs and outputs
            for each unit in the network.
        """        

        out = inp
        units_out = [out]
        nets = []
        for b, w in zip(self.biases[:-1], self.weights[:-1]):
            net = np.matmul(out, w.T) + b
            out = self.act.function(net)
            nets.append(net)
            units_out.append(out)

        # Last layer is linear for regression tasks and sigmoid for classification
        out = self.last_act.function(np.matmul(out, self.weights[-1].T) + self.biases[-1])
        nets.append(out)
        units_out.append(out)

        return units_out, nets, out

    
    def _backpropagation_batch(self, x, y, der):
        """Performs a backward pass by using the chain rule to compute the gradient
        for each weight and bias in the network for the current input/output samples.
        The computed gradient is dependend to the :der: parameter which specifies the
        function to use to compute the derivative.

        Parameters
        ----------
        x : np.ndarray
            Samples the will be used to estimate the current gradient values.
        y : np.ndarray
            Expected output for the :x: samples. Used to loss computation.
        der : function
            Function to be used to compute the derivative of the loss function
            w.r.t. each weight and bias in the network.

        Returns
        -------
        list, list
            Lists of np.ndarray containing for each layer the gradient matrix for each
            weight and bias in the network.
        """        

        nabla_b = [0]*(len(self.sizes)-1)
        nabla_w = [0]*(len(self.sizes)-1)

        # Forward computation
        units_out, nets, out = self._feedforward_batch(x)
        delta = 0

        # Backward pass
        for l in range(1, self.num_layers):
            if l == 1:
                # Backward pass - output unit
                delta = (out - y)
                delta = delta * self.last_act.derivative(nets[-1])
            else:
                # Backward pass - hidden unit
                delta = np.matmul(delta, self.weights[-l+1])
                delta = delta * der(nets[-l])

            nabla_b[-l] = delta.sum(axis=0)
            nabla_w[-l] = np.matmul(delta.T, units_out[-l-1])

        return nabla_b, nabla_w


    def train(self, optimizer, training_data, epochs, eta, batch_size=None, test_data=None):
        self.optimizer = OPTIMIZERS[optimizer](training_data, epochs, eta, batch_size=batch_size, test_data=test_data)
        self.optimizer.optimize(self)


    def evaluate(self, test_data, train_data):
        """Evaluates the performances of the Network in the current state,
        propagating the test examples through the network via a complete feedforward
        step. It evaluates the performance using the associated loss for this Network.
        Typical scores are MeanSquaredError and AccuracyScore.

        Parameters
        ----------
        test_data : tuple
            Couple of np.ndarray representing test samples and associated outputs. Used
            to test the generalization capabilities of the network.
        train_data : tuple
            Couple of np.ndarray representing training samples and associated outputs. Used
            to train the network and update weights accordingly.

        Returns
        -------
        tuple, list, list
            (score_test, score_train) representing the achieved score for both test and training samples;
            preds_train list of predictions for the training samples :train_data:;
            preds_test list of predictions for the test samples :test_data:
        """        

        score_test = []
        score_train = []
            
        preds_test = self._predict(test_data[0])
        truth_test = test_data[1]

        preds_train = self._predict(train_data[0])
        truth_train = train_data[1]
        
        score_test.append(self.loss.loss(truth_test, preds_test))
        score_train.append(self.loss.loss(truth_train, preds_train))

        self.val_scores.append(score_test[-1])
        self.train_scores.append(score_train[-1])

        return (score_test, score_train), preds_train, preds_test


    @abstractmethod
    def best_score(self):
        """Returns the best score achieved during the training of the current network.
        """        
        pass


    @abstractmethod
    def _predict(self, data):
        pass


    def plot_score(self, name):
        """Utility function, allows to build a plot of the scores achieved during training
        for the validation set and the training set.

        Parameters
        ----------
        name : string
            Prefix name for the plot file.
        """        

        plt.plot(self.val_scores, '--', label='Validation loss')
        plt.plot(self.train_scores, '--', label='Training loss')
        plt.legend(loc='best')
        plt.xlabel ('Epochs')
        plt.ylabel ('Loss')
        plt.title ('Loss NN CUP dataset')
        plt.draw()

        plt.savefig(f"src/NN/res/{name}ep{self.epochs}s{self.sizes}b{self.batch_size}e{self.eta}lmbda{self.lmbda}m{self.momentum}.png")
        plt.clf()

    
    def plot_grad(self, name):
        """Utility function, allows to build a plot of the gradient values achieved during
        training of the current Network.

        Parameters
        ----------
        name : string
            Prefix name for the plot file.
        """        

        plt.plot(self.grad_est_per_epoch, '--', label='Validation loss')
        plt.legend(loc='best')
        plt.xlabel ('Epochs')
        plt.ylabel ('Gradient\'s norm')
        plt.title ('Gradient norm estimate')
        plt.draw()

        plt.savefig(f"src/NN/res/grads/{name}ep{self.epochs}s{self.sizes}b{self.batch_size}e{self.eta}lmbda{self.lmbda}m{self.momentum}.png")
        plt.clf()



class NC(Network): 

    def __init__(self, sizes, seed, activation='sigmoid', lmbda=0.0, momentum=0.5, debug=True):
        """Neural Network implementation for classification tasks with sigmoid activation function
        in the output layer. 

        Parameters
        ----------
        sizes : tuple
            Tuple (i, l1, l2, ..., ln, o) containig the number of units for each layer,
            where the first and last elements represents, respectively, the input layer
            and the output layer units.
        seed : int
            seed for random number generator used for initializing this network weights
            and biases. Needed for reproducibility.
        activation : function, optional
            specifies which activation function to use for the hidden layers of the network,
            by default 'sigmoid'
        lmbda : int, optional
            l2 regularization coefficient, by default 0.0
        momentum : int, optional
            momentum coefficient, by default 0.0
        debug : bool, optional
            debugging flag, by default True
        """        

        super().__init__(sizes, seed, activation, lmbda, momentum, debug)

        # Defines the behavior of the last layer of the network
        self.last_act = Sigmoid
        self.loss = AccuracyScore


    def best_score(self):
        """Returns the best score achieved during the training of the
        current Network.

        Returns
        -------
        tuple
            Couple of values representing the best score for validation and training sets.
        """        

        best_score = ()
        if len(self.val_scores) > 0:
            best_score = (np.max(self.val_scores), np.max(self.train_scores))

        return best_score


    def _predict(self, data):
        """Performs a feedforward pass through the network for the given :data: samples,
        returns the classification values for each sample.

        Parameters
        ----------
        data : np.ndarray
            Samples to use for prediction using the current configuration of the Network.

        Returns
        -------
        np.ndarray
            Binary classification prediction values for the given :data: samples.
        """        
        return self._feedforward_batch(data)[2] >= 0.5



class NR(Network):

    def __init__(self, sizes, seed, activation='sigmoid', lmbda=0.0, momentum=0.5, debug=True):
        """Neural Network implementation for regression tasks with linear activation function
        in the output layer. 

        Parameters
        ----------
        sizes : tuple
            Tuple (i, l1, l2, ..., ln, o) containig the number of units for each layer,
            where the first and last elements represents, respectively, the input layer
            and the output layer units.
        seed : int
            seed for random number generator used for initializing this network weights
            and biases. Needed for reproducibility.
        activation : function, optional
            specifies which activation function to use for the hidden layers of the network,
            by default 'sigmoid'
        lmbda : int, optional
            l2 regularization coefficient, by default 0.0
        momentum : int, optional
            momentum coefficient, by default 0.0
        debug : bool, optional
            debugging flag, by default True
        """ 

        super().__init__(sizes, seed, activation, lmbda, momentum, debug)

        # Defines the behavior of the last layer of the network
        self.last_act = Linear
        self.loss = MeanSquaredError


    def best_score(self):
        """Returns the best score achieved during the training of the
        current Network.

        Returns
        -------
        tuple
            Couple of values representing the best score for validation and training sets.
        """ 

        best_score = ()
        if len(self.val_scores) > 0:
            best_score = (np.min(self.val_scores), np.min(self.train_scores))

        return best_score


    def _predict(self, data):
        """Performs a feedforward pass through the network for the given :data: samples,
        returns the values for each sample for the regression task.

        Parameters
        ----------
        data : np.ndarray
            Samples to use for prediction using the current configuration of the Network.

        Returns
        -------
        np.ndarray
            Regression prediction values for the given :data: samples.
        """

        return self._feedforward_batch(data)[2]