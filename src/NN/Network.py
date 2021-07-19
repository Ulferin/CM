# This class represents the Network used in the CM project and it is
# implemented from scratch following the advices taken during the course
# of ML



from abc import ABCMeta, abstractmethod

import numpy as np
from numpy.random import default_rng

from sklearn.utils import shuffle
from sklearn.metrics import r2_score

from matplotlib import pyplot as plt

from src.NN.ActivationFunctions import ReLU, Sigmoid, LeakyReLU, Linear
from src.NN.metrics import mean_squared_error, accuracy_score
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
        following the LeCun uniform initializer formula. Various hyperparameters can be specified,
        like momentum and regularization coefficients.

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
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.act = ACTIVATIONS[activation]
        self.momentum = momentum
        self.lmbda = lmbda

        self.last_act = None            # Must be defined by subclassing the Network
        self.grad_est = None
        self.grad_est_per_epoch = []

        self.biases = [rng.normal(0,0.5,l) for l in sizes[1:]]
        self.weights = [rng.uniform(-np.sqrt(3/x), np.sqrt(3/x), (y,x)) for x, y in zip(sizes[:-1], sizes[1:])]
        self.wvelocities = [np.zeros_like(weight) for weight in self.weights]
        self.bvelocities = [np.zeros_like(bias) for bias in self.biases]

        self.val_scores = []
        self.train_scores = []
        self.val_loss = []
        self.train_loss = []

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

    
    def _backpropagation_batch(self, x, y):
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
                delta = delta * self.der(nets[-l])

            nabla_b[-l] = delta.sum(axis=0)
            nabla_w[-l] = np.matmul(delta.T, units_out[-l-1])

        return nabla_b, nabla_w


    def _create_batches(self, batches, training_data):
        """Creates a list of mini-batches that will be used during optimization.

        Parameters
        ----------
        batches : int
            number of batches that will be created from the current :training_data:.
        training_data : np.ndarray
            training data samples used to build the batches used for training.

        Returns
        -------
        list
            a list of mini-batches created with the specified :batches: number from the :training_data: samples.
        """                      
        mini_batches = []

        if self.batch_size is not None:
            for b in range(batches):
                start = b * self.batch_size
                end = (b+1) * self.batch_size
                mini_batches.append((training_data[0][start:end], training_data[1][start:end]))
            
            # Add remaining data as last batch
            # (it may have different size than previous batches, up to |batch|-1 more elements)
            mini_batches.append((training_data[0][b*self.batch_size:], training_data[1][b*self.batch_size:]))
        else:
            mini_batches.append((training_data[0], training_data[1]))

        return mini_batches


    def _update_batches(self, training_data):
        """Creates batches and updates the Neural Network weights and biases by performing
        updates using the optimizer associated to the current network.

        Parameters
        ----------
        training_data : tuple
            tuple of np.ndarray containing the training samples and the associated expected outputs.
        """               
        # training_data = shuffle(training_data[0], training_data[1], random_state=42)
        mini_batches = self._create_batches(self.batches, training_data)
        self.grad_est = 0

        for mini_batch in mini_batches:
            nabla_b, nabla_w = self._compute_grad(mini_batch)
            self.optimizer.update_mini_batch(self, nabla_b, nabla_w, len(mini_batch[0]))
            self.grad_est += self.ngrad


    def _compute_grad(self, mini_batch):
        """Computes the gradient values and norm for the current :mini_batch: samples by
        using the backpropagation method implemented by the Neural Network object.

        Parameters
        ----------
        mini_batch : np.ndarray
            mini-batch samples for which to compute the gradient.

        Returns
        -------
        np.ndarray, np.ndarray
            Couple of np.ndarray indicating the gradient values for both
            weights and biases.
        """        

        nabla_b, nabla_w = self._backpropagation_batch(mini_batch[0], mini_batch[1])
        self.ngrad = np.linalg.norm(np.hstack([el.ravel() for el in nabla_w + nabla_b])/len(mini_batch[0]))

        return nabla_b, nabla_w


    def train(self, optimizer, training_data, epochs, eta, eps=1e-5, batch_size=None, test_data=None):
        """Trains the neural network on :training_data: sample for a given number of :epochs:
        by fine-tuning the weights and biases by using the update rules relative to
        the provided :optimizer:. The way updates are performed is also determined by the
        configurations relative to :batch_size: and :eta: parameters.

        Parameters
        ----------
        optimizer : string
            Specifies which optimizer technique should be used for updates.
        training_data : tuple
            Tuple containing the training samples to use for training the neural network and the
            associated expected outputs for each sample.
        epochs : int
            Max number of iterations for the training of the network.
        eta : float
            Learning rate parameter if SGD optimizer is used, starting step for SGM.
        batch_size : int, optional
            If specified determines the size of the batches used to train the network, by default None
        test_data : tuple, optional
            If provided, test samples and expected outputs are used to evaluate the performance
            of the current network at each epoch of training, by default None
        """        
        
        self.der = self.act.derivative if optimizer == 'SGD' else self.act.subgrad

        self.batch_size = batch_size
        self.eta = eta
        self.epochs = epochs
        self.training_size = len(training_data[0])
        self.batches = int(self.training_size/batch_size) if batch_size is not None else 1

        self.training_data = training_data
        self.test_data = test_data

        self.optimizer = OPTIMIZERS[optimizer](training_data, epochs, eta, eps=eps, batch_size=batch_size, test_data=test_data)

        for e in range(1, self.epochs+1):
            self._update_batches(training_data)

            # Compute current gradient estimate
            self.grad_est = self.grad_est/self.batches
            self.grad_est_per_epoch.append(self.grad_est)
            self.evaluate(e)

            self.score = self.train_loss[-1]

            if self.optimizer.iteration_end(e, self):
                print("Reached desired precision in gradient norm, stopping.")
                break


    def evaluate(self, e):
        """Returns statistics for the current epoch if test data are provided while training the network.
        It prints the current epoch, gradient norm for convergence analysis and the current score computed
        as loss value.

        Parameters
        ----------
        e : int
            current epoch.

        Returns
        -------
        float
            latest score achieved during evaluation over training data.
        """              

        if self.test_data is not None:
            self._evaluate(self.training_data, self.test_data)
            print(f"{e:<7} || Gradient norm: {self.ngrad:7.5e} || Loss: {self.val_loss[-1]:7.5e}, {self.train_loss[-1]:7.5e} || Score: {self.val_scores[-1]:5.3g}, {self.train_scores[-1]:<5.3g}")
        else:
            self._evaluate(self.training_data)
            print(f"{e:<7} || Gradient norm: {self.ngrad:7.5e} || Loss: {self.train_loss[-1]:7.5e} || Score: {self.train_scores[-1]:<5.3g}")


    def _evaluate(self, train_data, test_data=None):
        """Evaluates the performances of the Network in the current state,
        propagating the test and training examples through the network via a complete feedforward
        step. It evaluates the performance using the associated loss for this Network.
        Typical scores are mean_squared_error and accuracy_score.

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
        if test_data:            
            preds_test = self._predict(test_data[0])
            truth_test = test_data[1]

            self.val_loss.append(self.loss(truth_test, self._feedforward_batch(test_data[0])[2]))
            self.val_scores.append(self.scoring(truth_test, preds_test))

        preds_train = self._predict(train_data[0])
        truth_train = train_data[1]

        self.train_loss.append(self.loss(truth_train, self._feedforward_batch(train_data[0])[2]))
        self.train_scores.append(self.scoring(truth_train, preds_train))


    def best_score(self, name, save=False):
        """Returns the best score achieved during the training of the
        current Network.

        Returns
        -------
        tuple
            Couple of values representing the best score for validation and training sets.
        """        

        best_score = ()
        if len(self.val_scores) > 0:
            idx = np.argmax(self.val_scores)
            best_score = (self.val_scores[idx], self.train_scores[idx])
            idx = np.argmin(self.val_loss)
            best_loss = (self.val_loss[idx], self.train_loss[idx])

        score_file = open(f"src/NN/res/scores/{name}.txt", 'a')
        
        stats = f"ep: {self.epochs:<7} | s: {self.sizes[1:-1]} | b: {self.batch_size} | e:{self.eta:5} | lmbda:{self.lmbda:5} | m:{self.momentum:5}\n"\
                f"Grad: {self.ngrad:7.5e} | Loss: {best_loss[0]:7.5e}, {best_loss[1]:7.5e} | Score: {best_score[0]:5.3g}, {best_score[1]:<5.3g}\n\n"

        score_file.write(stats)
        score_file.close()

        return (best_score, best_loss)


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
        self.loss = mean_squared_error
        self.scoring = accuracy_score


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
        self.loss = mean_squared_error
        self.scoring = r2_score


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