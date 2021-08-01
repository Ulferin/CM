# This class represents the Network used in the CM project and it is
# implemented from scratch following the advices taken during the course
# of ML



from abc import ABCMeta, abstractmethod

import numpy as np
from numpy.random import default_rng
from sklearn.base import BaseEstimator

from sklearn.utils import shuffle
from sklearn.metrics import r2_score

from matplotlib import pyplot as plt

from src.NN.ActivationFunctions import ReLU, Sigmoid, LeakyReLU, Linear
from src.NN.metrics import mean_squared_error, accuracy_score
from src.NN.optimizers import SGD, SGM
from src.NN.utils import end_time

from datetime import datetime as dt
from time import sleep


ACTIVATIONS = {
    'relu': ReLU,
    'Lrelu':LeakyReLU,
    'sigmoid': Sigmoid
}

OPTIMIZERS = {
    'SGD': SGD,
    'SGM': SGM
}


class Network(BaseEstimator, metaclass=ABCMeta):
    """Abstract class representing a standard Neural Network, also known as Multilayer Perceptron.
    It allows to build a network for both classification and regression tasks by using the
    preferred optimization technique between sub-gradient method and stochastic gradient descent.
    Must be extended to specify which kind of task the network should solve.
    """    
    
    @abstractmethod
    def __init__(self, sizes=None, optimizer='SGD', seed=0, epochs=1000, eta=0.01, activation='Lrelu', lmbda=0.0, momentum=0.0, nesterov=False, eps=1e-5, batch_size=None, debug=False,):
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

        self.rng = default_rng(seed)     # needed for reproducibility

        # Network hyperparameters
        self.batch_size = batch_size
        self.eta = eta
        self.eps = eps
        self.epochs = epochs
        self.optimizer = optimizer
        self.seed = seed
        self.debug = debug
        self.momentum = momentum
        self.nesterov = nesterov
        self.lmbda = lmbda
        self.sizes = sizes
        self.activation = activation
        self.last_act = None            # Must be defined by subclassing the Network

        # Performance attributes
        self.grad_est = []
        self.grad_est_per_epoch = []
        self.val_scores = []
        self.train_scores = []
        self.val_loss = []
        self.train_loss = []

        # Execution Statistics
        self.evaluate_avg = [0, 0]
        self.backprop_avg = [0, 0]
        self.feedforward_avg = [0, 0]
        self.epochs_time = []
        self.total_time = 0
        self.update_avg = 0


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
        start = dt.now()      

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

        end = end_time(start)
        self.feedforward_avg[0] += 1
        self.feedforward_avg[1] += end.seconds*1000 + end.microseconds/1000

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


        nabla_b = [0]*(len(self._sizes)-1)
        nabla_w = [0]*(len(self._sizes)-1)

        # Forward computation
        units_out, nets, out = self._feedforward_batch(x)
        delta = 0

        start = dt.now()

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

        end = end_time(start)
        self.backprop_avg[0] += 1
        self.backprop_avg[1] += end.seconds*1000 + end.microseconds/1000

        return nabla_b, nabla_w


    def _create_batches(self):
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

        if self.batch_size < self.training_size:
            for b in range(self.batches):
                start = b * self.batch_size
                end = (b+1) * self.batch_size
                mini_batches.append((self.X[start:end], self.y[start:end]))
            
            # Add remaining data as last batch
            # (it may have different size than previous batches, up to |batch|-1 more elements)
            last = (self.X[self.batches*self.batch_size:], self.y[self.batches*self.batch_size:])
            if len(last[0]) > 0: mini_batches.append(last)
        else:
            mini_batches.append((self.X, self.y))

        return mini_batches


    def _update_batches(self):
        """Creates batches and updates the Neural Network weights and biases by performing
        updates using the optimizer associated to the current network.

        Parameters
        ----------
        training_data : tuple
            tuple of np.ndarray containing the training samples and the associated expected outputs.
        """               
        # training_data = shuffle(training_data[0], training_data[1], random_state=42)
        mini_batches = self._create_batches()
        self.grad_est = []
        self.num_batches = len(mini_batches)

        for mini_batch in mini_batches:
            self.opti.update_mini_batch(self, mini_batch)
            self.grad_est.append(self.ngrad)


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


    def fit(self, X, y, test_data=None):
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
            Determines the size of the batches used to train the network, by default None
        test_data : tuple, optional
            If provided, test samples and expected outputs are used to evaluate the performance
            of the current network at each epoch of training, by default None
        """     


        # Initialize batch size w.r.t. training data
        self.X = X
        self.y = y
        self.test_data = test_data   
        self.training_size = len(self.X)
        self.batch_size = self.batch_size if self.batch_size is not None and self.batch_size > 0\
                                            else self.training_size
        self.batches = int(self.training_size/self.batch_size)

        # Set up activation function and optimizer
        self.act = ACTIVATIONS[self.activation]
        self.opti = OPTIMIZERS[self.optimizer](self.eta, eps=self.eps)
        self.der = self.act.derivative if self.optimizer == 'SGD' else self.act.subgrad

        # Set up input/output units
        self._sizes = self.sizes.copy()
        self._sizes.insert(0, self.X.shape[1])
        self._sizes.append(1 if len(self.y.shape) == 1 else self.y.shape[1])
        self.num_layers = len(self._sizes)

        # Initialize network parameters
        self.biases = [self.rng.normal(0,0.5,l) for l in self._sizes[1:]]
        self.weights = [self.rng.uniform(-np.sqrt(3/x), np.sqrt(3/x), (y,x)) for x, y in zip(self._sizes[:-1], self._sizes[1:])]
        self.wvelocities = [np.zeros_like(weight) for weight in self.weights]
        self.bvelocities = [np.zeros_like(bias) for bias in self.biases]

        start = dt.now()
        self.fitted = True

        try:
            for e in range(1, self.epochs+1):
                s = dt.now()
                self._update_batches()
                en = end_time(s)
                self.update_avg = en.seconds*1000 + en.microseconds/1000

                # Compute current gradient estimate
                self.grad_est_per_epoch.append(np.average(self.grad_est))
                self.evaluate(e)

                self.score = self.train_loss[-1]
                epoch_time = end_time(start)
                self.epochs_time.append(epoch_time.seconds*1000 + epoch_time.microseconds/1000)

                if self.opti.iteration_end(e, self):
                    print("Reached desired precision in gradient norm, stopping.")
                    break
        except ValueError:
            print("Computation did not finish due to NaN.")
            self.fitted = False
        finally:
            end = end_time(start)
            self.total_time = end.seconds*1000 + end.microseconds/1000
            return self


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

        start = dt.now()

        if self.test_data is not None:
            self._evaluate((self.X, self.y), self.test_data)
            if self.debug: print(f"{e:<7} || Gradient norm: {self.ngrad:7.5e} || Loss: {self.val_loss[-1]:7.5e}, {self.train_loss[-1]:7.5e} || Score: {self.val_scores[-1]:5.3g}, {self.train_scores[-1]:<5.3g}")
        else:
            self._evaluate((self.X, self.y))
            if self.debug: print(f"{e:<7} || Gradient norm: {self.ngrad:7.5e} || Loss: {self.train_loss[-1]:7.5e} || Score: {self.train_scores[-1]:<5.3g}")

        end = end_time(start)
        self.evaluate_avg[1] += end.seconds*1000 + end.microseconds/1000
        self.evaluate_avg[0] +=1


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
            preds_test = self.predict(test_data[0])
            truth_test = test_data[1]

            self.val_loss.append(self.loss(truth_test, self.last_pred))
            self.val_scores.append(self.scoring(truth_test, preds_test))

        preds_train = self.predict(train_data[0])
        truth_train = train_data[1]

        self.train_loss.append(self.loss(truth_train, self.last_pred))
        self.train_scores.append(self.scoring(truth_train, preds_train))


    def best_score(self, name="", save=False):
        """Returns the best score achieved during the training of the
        current Network.

        Returns
        -------
        tuple
            Couple of values representing the best score for validation and training sets.
        """        

        if not self.fitted:
            return 'This model is not fitted yet.\n\n'

        best_score = ()
        if len(self.val_scores) > 0:
            idx = np.argmax(self.val_scores)
            best_score = (self.val_scores[idx], self.train_scores[idx])
            idx = np.argmin(self.val_loss)
            best_loss = (self.val_loss[idx], self.train_loss[idx])
        else:
            best_score = (-1, np.max(self.train_scores))
            best_loss = (-1, np.min(self.train_loss))

        stats = f"ep: {self.epochs:<7} | s: {self.sizes} | b: {self.batch_size} | e:{self.eta:5} | lmbda:{self.lmbda:5} | m:{self.momentum:5} | nesterov: {self.nesterov}\n"\
                f"Grad: {self.ngrad:7.5e} | Loss: {best_loss[0]:7.5e}, {best_loss[1]:7.5e} | Score: {best_score[0]:5.3g}, {best_score[1]:<5.3g}\n"\
                f"ended in: {self.total_time}, avg per ep: {self.total_time/self.epochs}\n"\
                f"total update: {self.update_avg}, avg updt: {self.update_avg/self.epochs}\n"\
                f"total ff: {self.feedforward_avg[0]}, total ff time: {self.feedforward_avg[1]}, avg ff: {self.feedforward_avg[1]/self.feedforward_avg[0]}\n"\
                f"total bp: {self.backprop_avg[0]}, total bp time: {self.backprop_avg[1]}, avg bp: {self.backprop_avg[1]/self.backprop_avg[0]}\n"\
                f"total ev: {self.evaluate_avg[0]}, total ev time: {self.evaluate_avg[1]}, avg ev: {self.evaluate_avg[1]/self.evaluate_avg[0]}\n\n"

        if save:
            file_path = f"src/NN/res/best_scores/stats.txt"
            with open(file_path, 'a') as f:
                f.write(f"{name}\n")
                f.write(stats)

        return stats


    @abstractmethod
    def predict(self, data):
        pass


    def plot_results(self, name, score=False, save=False, time=False):
        """Utility function, allows to build a plot of the scores achieved during training
        for the validation set and the training set.

        Parameters
        ----------
        name : string
            Prefix name for the plot file.
        """        

        # Conditional configuration
        x_label = 'Execution Time' if time else 'Epochs'
        folder = 'scores' if score else 'losses'
        sub_folder = 'time' if time else 'epochs'

        if self.test_data is not None:
            val_res = self.val_scores if score else self.val_loss
        train_res = self.train_scores if score else self.train_loss
        x = self.epochs_time if time else list(range(len(train_res)))

        if self.test_data is not None:
            plt.plot(x, val_res, '--', label='Validation loss')
        plt.plot(x, train_res, '--', label='Training loss')
        
        plt.xlabel(x_label)
        plt.ylabel ('Loss')
        
        plt.legend(loc='best')
        plt.title ('Loss NN CUP dataset')
        plt.draw()


        if save: plt.savefig(f"src/NN/res/{folder}/{sub_folder}/{name}ep{self.epochs}s{self.sizes}b{self.batch_size}e{self.eta}lmbda{self.lmbda}m{self.momentum}.png")
        else: plt.show()
        plt.clf()

    
    def plot_grad(self, name, save=False):
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
        plt.yscale('log')
        plt.draw()

        if save: plt.savefig(f"src/NN/res/grads/{name}ep{self.epochs}s{self.sizes}b{self.batch_size}e{self.eta}lmbda{self.lmbda}m{self.momentum}.png")
        else: plt.show()
        plt.clf()



class NC(Network, BaseEstimator): 
    def __init__(self, sizes=None, optimizer='SGD', seed=0, epochs=300, eta=0.1, activation='Lrelu', lmbda=0.0001, momentum=0.5, nesterov=False, eps=1e-5, batch_size=10, debug=False):
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

        super().__init__(sizes=sizes,
                        optimizer=optimizer,
                        seed=seed,
                        epochs=epochs,
                        eta=eta,
                        activation=activation,
                        lmbda=lmbda,
                        momentum=momentum,
                        nesterov=nesterov,
                        debug=debug,
                        eps=eps,
                        batch_size=batch_size)

        # Defines the behavior of the last layer of the network
        self.last_act = Sigmoid
        self.loss = mean_squared_error
        self.scoring = accuracy_score
        self.last_pred = None


    def predict(self, data):
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
        self.last_pred = self._feedforward_batch(data)[2]

        return  self.last_pred >= 0.5



class NR(Network, BaseEstimator):
    def __init__(self, sizes=None, optimizer='SGD', seed=0, epochs=1000, eta=0.01, activation='Lrelu', lmbda=0.0, momentum=0.0, nesterov=False, eps=1e-5, batch_size=None, debug=False):
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
        super().__init__(sizes=sizes,
                        optimizer=optimizer,
                        seed=seed,
                        epochs=epochs,
                        eta=eta,
                        activation=activation,
                        lmbda=lmbda,
                        momentum=momentum,
                        nesterov=nesterov,
                        debug=debug,
                        eps=eps,
                        batch_size=batch_size)


        # Defines the behavior of the last layer of the network
        self.last_act = Linear
        self.loss = mean_squared_error
        self.scoring = r2_score
        self.last_pred = None


    def predict(self, data):
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
        self.last_pred = self._feedforward_batch(data)[2]
        return self.last_pred