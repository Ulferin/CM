# This class represents the Network used in the CM project and it is
# implemented from scratch following the advices taken during the course
# of ML

from datetime import datetime as dt
from abc import ABCMeta, abstractmethod

import numpy as np
from numpy.random import default_rng

from sklearn.base import BaseEstimator
from sklearn.utils import shuffle
from sklearn.metrics import r2_score

from matplotlib import pyplot as plt

from src.NN.ActivationFunctions import ReLU, Sigmoid, LeakyReLU, Linear
from src.NN.metrics import mean_squared_error, accuracy_score
from src.NN.optimizers import SGD, Adam
from src.utils import end_time


ACTIVATIONS = {
    'relu': ReLU,
    'Lrelu':LeakyReLU,
    'sigmoid': Sigmoid
}

OPTIMIZERS = {
    'SGD': SGD,
    'Adam': Adam
}


class Network(BaseEstimator, metaclass=ABCMeta):
    """Abstract class representing a standard Neural Network, also known as
    Multilayer Perceptron. It allows to build a network for both classification
    and regression tasks by using the preferred optimization technique between
    sub-gradient method and stochastic gradient descent. Must be extended to 
    specify which kind of task the network should solve.
    """    
    
    @abstractmethod
    def __init__(self,
                sizes=None, optimizer='SGD', seed=0, epochs=1000, eta=0.01,
                activation='Lrelu', lmbda=0.0, momentum=0.0, nesterov=False,
                eps=1e-5, batch_size=None, debug=False, beta1=0.9, beta2=0.999):            

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
        self.last_act = None       # Must be defined by subclassing the Network

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

        # Initialize optimizer
        optimizer_params = {
            'eta': eta,
            'eps': eps,
            'lmbda': lmbda
        }
        if optimizer == 'SGD':
            optimizer_params['momentum'] = momentum
            optimizer_params['nesterov'] = nesterov
        elif optimizer == 'Adam':
            optimizer_params['beta1'] = beta1
            optimizer_params['beta2'] = beta2

        self.opti = OPTIMIZERS[self.optimizer](**optimizer_params)


    def _feedforward_batch(self, inp):
        """Performs a feedforward pass through the network and returns the
        related output.

        Parameters
        ----------
        inp : np.ndarray
            Network input used to perform the feedforward pass.

        Returns
        -------
        units_out : np.ndarray
            output related to the current input :inp:

        nets : list
            list of unit's input for each layer of the network.

        out : list
            list of unit's output for each layer of the network. 
        """  
        start = dt.now()      

        out = inp
        units_out = [out]
        nets = []

        # Forward pass
        for b, w in zip(self.biases[:-1], self.weights[:-1]):
            net = np.matmul(out, w.T) + b
            out = self.act.function(net)
            nets.append(net)
            units_out.append(out)

        # Last layer is linear for regression and sigmoid for classification
        net = np.matmul(out,self.weights[-1].T) + self.biases[-1]
        out = self.last_act.function(net)

        nets.append(net)
        units_out.append(out)

        end = end_time(start)
        self.feedforward_avg[0] += 1
        self.feedforward_avg[1] += end

        return units_out, nets, out

    
    def _backpropagation_batch(self, x, y):
        """Performs a backward pass by using the chain rule to compute the
        gradient for each weight and bias in the network for the current
        input/output samples.

        Parameters
        ----------
        x : np.ndarray
            Samples that will be used to estimate the current gradient values.
        
        y : np.ndarray
            Expected output for the :x: samples. Used for loss computation.

        Returns
        -------
        nabla_b : list
            List of np.ndarray containing for each layer the gradients for
            each bias in the network.
        
        nabla_w : list
            List of np.ndarray containing for each layer the gradients for
            each weight in the network.
        """        
        delta = 0
        nabla_b = [0]*(len(self._sizes)-1)
        nabla_w = [0]*(len(self._sizes)-1)

        # Forward computation
        units_out, nets, out = self._feedforward_batch(x)

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

        # Computes execution statistics
        end = end_time(start)
        self.backprop_avg[0] += 1
        self.backprop_avg[1] += end

        return nabla_b, nabla_w


    def _create_batches(self):
        """Creates a list of mini-batches that will be used during optimization.

        Returns
        -------
        mini_batches : list
            a list of mini-batches
        """                      
        mini_batches = []

        if self.batch_size < self.training_size:
            for b in range(self.batches):
                start = b * self.batch_size
                end = (b+1) * self.batch_size
                mini_batches.append((self.X[start:end], self.y[start:end]))
            
            # Add remaining data as last batch
            # (it may have different size than previous batches)
            last = (
                self.X[self.batches*self.batch_size:],
                self.y[self.batches*self.batch_size:])

            if len(last[0]) > 0: mini_batches.append(last)
        else:
            mini_batches.append((self.X, self.y))

        return mini_batches


    def _update_batches(self):
        """Creates batches and updates the Neural Network weights and biases
        by performing updates using the optimizer associated to the current
        network.
        """               
        mini_batches = self._create_batches()
        self.grad_est = []
        self.num_batches = len(mini_batches)

        for mini_batch in mini_batches:
            params = self.weights + self.biases
            size = len(mini_batch[0])
            
            # Compute current gradient
            nabla_b, nabla_w = self._compute_grad(mini_batch)
            grads = nabla_w + nabla_b
            
            self.opti.update_parameters(params, grads, size)
            self.grad_est.append(self.ngrad)


    def _compute_grad(self, mini_batch):
        """Computes the gradient values and norm for the current :mini_batch:
        samples by using the backpropagation approach.

        Parameters
        ----------
        mini_batch : np.ndarray
            mini-batch samples for which to compute the gradient.

        Returns
        -------
        nabla_b : np.ndarray
            Gradients w.r.t. network biases.
        
        nabla_w : np.ndarray
            Gradients w.r.t network weights.
        """        

        nabla_b, nabla_w = self._backpropagation_batch(
                                    mini_batch[0],mini_batch[1])
        self.ngrad = np.linalg.norm(
                        np.hstack([el.ravel() for el in nabla_w + nabla_b])
                        / len(mini_batch[0]))

        return nabla_b, nabla_w


    def fit(self, X, y, test_data=None):   
        """Trains the neural network on (:X:, :y:) samples for a given
        number of epochs by fine-tuning the weights and biases by using the
        update rules relative to the provided optimizer. The way updates are
        performed is also determined by the configurations relative to 
        batch size and eta hyperparameters.

        Parameters
        ----------
        X : np.ndarray
            training samples to use to train the neural network.

        y : np.ndarray
            expected outputs for the given training samples :X:

        test_data : tuple, optional
            If provided, test samples and expected outputs are used to evaluate
            the instantaneous performance of the current network at each epoch
            of training, by default None

        Returns
        -------
        Object
            returns the fitted network object
        """

        # Initialize batch size w.r.t. training data
        self.X = X
        self.y = y
        self.test_data = test_data   
        self.training_size = len(self.X)
        self.batch_size = (
            self.batch_size
            if self.batch_size is not None and self.batch_size > 0
            else self.training_size)
        self.batches = int(self.training_size/self.batch_size)

        # Set up activation function and optimizer
        self.act = ACTIVATIONS[self.activation]
        # self.opti = OPTIMIZERS[self.optimizer](self.eta, eps=self.eps)
        self.der = self.act.derivative

        # Set up input/output units
        self._sizes = self.sizes.copy()
        self._sizes.insert(0, self.X.shape[1])
        self._sizes.append(1 if len(self.y.shape) == 1 else self.y.shape[1])
        self.num_layers = len(self._sizes)

        # Initialize network parameters
        self.biases = [np.array(self.rng.normal(0,0.5,l)) for l in self._sizes[1:]]
        self.weights = [
            np.array(self.rng.uniform(-np.sqrt(3/x), np.sqrt(3/x), (y,x)))
            for x, y in zip(self._sizes[:-1], self._sizes[1:])]

        start = dt.now()
        self.fitted = True
        try:
            for e in range(1, self.epochs+1):
                s = dt.now()
                self._update_batches()
                en = end_time(s)
                self.update_avg += en

                # Compute current gradient estimate
                self.grad_est_per_epoch.append(np.average(self.grad_est))
                self.evaluate(e)

                self.score = self.train_loss[-1]
                iteration_end = self.opti.iteration_end(self.ngrad)
                
                epoch_time = end_time(start)
                self.epochs_time.append(epoch_time)

                if iteration_end:
                    print("Reached desired precision in gradient norm,stopping.")
                    break
        except ValueError:
            self.fitted = False
        finally:
            end = end_time(start)
            self.total_time = end
            return self


    def evaluate(self, e):
        """Returns statistics for the current epoch if test data are provided
        while training the network. It prints the current epoch, gradient norm
        for convergence analysis and the current score computed as loss value.

        Parameters
        ----------
        e : int
            current epoch.
        """              

        start = dt.now()

        if self.test_data is not None:
            self._evaluate((self.X, self.y), self.test_data)
            if self.debug: print(
                f"{e:<7} || Gradient norm: {self.ngrad:7.5e} || "
                f"Loss: {self.val_loss[-1]:7.5e}, {self.train_loss[-1]:7.5e} ||"
                f" Score: {self.val_scores[-1]:5.3g}, "
                f"{self.train_scores[-1]:<5.3g}")
        else:
            self._evaluate((self.X, self.y))
            if self.debug: print(
                f"{e:<7} || Gradient norm: {self.ngrad:7.5e} || "
                f"Loss: {self.train_loss[-1]:7.5e} || "
                f"Score: {self.train_scores[-1]:<5.3g}")

        end = end_time(start)
        self.evaluate_avg[0] += 1
        self.evaluate_avg[1] += end


    def _evaluate(self, train_data, test_data=None):
        """Evaluates the performances of the Network in the current state,
        propagating the test and training examples through the network via a
        complete feedforward step. It evaluates the performance using the
        associated loss for this Network.

        Parameters
        ----------
        train_data : tuple
            Couple of np.ndarray representing training samples and associated
            outputs. Used to compute performance on training samples.

        test_data : tuple
            Couple of np.ndarray representing test samples and associated
            outputs. Used to test the generalization capabilities of the network.
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
        """Returns performance statistics related to achieved performances
        and gradient norm.

        Parameters
        ----------
        name : str, optional
            Only used when save=True, represents the name to use to save
            the statistics on a file, by default ""
        
        save : bool, optional
            specifies whether to create a file with the computed statistics,
            by default False

        Returns
        -------
        stats : string
            statistics computed during the network training.
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

        stats = (
            f"ep: {self.epochs:<7} | s: {self.sizes} | b: {self.batch_size} | "
            f"e:{self.eta:5} | lmbda:{self.lmbda:5} | m:{self.momentum:5} | "
            f"nesterov: {self.nesterov}\n"

            f"Grad: {self.ngrad:7.5e} | "
            f"Loss: {best_loss[0]:7.5e}, {best_loss[1]:7.5e} | "
            f"Score: {best_score[0]:5.3g}, {best_score[1]:<5.3g}\n"

            f"ended in: {self.total_time}, "
            f"avg per ep: {self.total_time/self.epochs}\n"

            f"total update: {self.update_avg}, "
            f"avg updt: {self.update_avg/self.epochs}\n"

            f"total ff: {self.feedforward_avg[0]}, "
            f"total ff time: {self.feedforward_avg[1]}, "
            f"avg ff: {self.feedforward_avg[1]/self.feedforward_avg[0]}\n"

            f"total bp: {self.backprop_avg[0]}, "
            f"total bp time: {self.backprop_avg[1]}, "
            f"avg bp: {self.backprop_avg[1]/self.backprop_avg[0]}\n"

            f"total ev: {self.evaluate_avg[0]}, "
            f"total ev time: {self.evaluate_avg[1]}, "
            f"avg ev: {self.evaluate_avg[1]/self.evaluate_avg[0]}\n\n")

        if save:
            file_path = f"src/NN/res/stats/stats.txt"
            with open(file_path, 'a') as f:
                f.write(f"{name}\n")
                f.write(stats)

        return stats


    @abstractmethod
    def predict(self, data):
        pass


    def plot_results(self, name, score=False, save=False, time=False, log=False):
        """Builds a plot of the scores achieved during training for the
        validation set and the training set.

        Parameters
        ----------
        name : string
            prefix file name to use when saving the plot.
        
        score : bool, optional
            whether to use the score instead of the loss, by default False
        
        save : bool, optional
            whether to save the plot on the file specified with the name
            parameter, by default False
        
        time : bool, optional
            whether build plots according to number of epochs or time of
            execution, by default False
        
        log : bool, optional
            whether to use a logplot or not, by default False
        """     

        if not self.fitted:
            return 'This model is not fitted yet.\n\n'

        # Conditional configuration
        x_label = 'Execution Time' if time else 'Epochs'
        curve_type = 'Loss' if not score else 'Score'

        if self.test_data is not None:
            val_res = self.val_scores if score else self.val_loss
        train_res = self.train_scores if score else self.train_loss
        x = self.epochs_time if time else list(range(len(train_res)))

        if self.test_data is not None:
            plt.plot(x, val_res, '--', label='Validation loss')
        plt.plot(x, train_res, '--', label='Training loss')
        
        plt.xlabel(x_label)
        plt.ylabel (curve_type)
        if log: plt.yscale('log')
        
        plt.legend(loc='best')
        plt.title (f'{curve_type} {self.optimizer}')
        plt.draw()


        if save:
            plt.savefig(
                f"src/NN/res/stats/{name}"
                f"ep{self.epochs}s{self.sizes}b{self.batch_size}e{self.eta}"
                f"lmbda{self.lmbda}m{self.momentum}.png")
        else:
            plt.show()
        plt.clf()

    
    def plot_grad(self, name, save=False, time=False):
        """Builds a plot of the gradient values achieved during training of the
        current network.

        Parameters
        ----------
        name : string
            prefix file name for the plot file, only used when save=True.
        
        save : bool, optional
            whether to save the plot on file or not, by default False
        
        time : bool, optional
            whether to plot gradient w.r.t. epochs or execution time,
            by default False
        """  

        if not self.fitted:
            return 'This model is not fitted yet.\n\n'

        x = self.epochs_time if time else list(range(len(self.epochs_time)))
        x_label = 'Execution Time' if time else 'Epochs'

        plt.plot(x, self.grad_est_per_epoch, '--', label='')
#         plt.legend(loc='best')
        plt.xlabel (x_label)
        plt.ylabel ('Gradient\'s norm')
        plt.title ('Gradient norm estimate')
        plt.yscale('log')
        plt.draw()

        if save:
            plt.savefig(
                f"src/NN/res/stats/{name}ep{self.epochs}s{self.sizes}"
                f"b{self.batch_size}e{self.eta}lmbda{self.lmbda}"
                f"m{self.momentum}.png")
        else:
            plt.show()
        plt.clf()



class NC(Network, BaseEstimator): 
    def __init__(
        self, sizes=None, optimizer='SGD', seed=0, epochs=300, eta=0.1,
        activation='Lrelu', lmbda=0.0001, momentum=0.5, nesterov=False,
        eps=1e-5, batch_size=None, debug=False):
        """Initializes the network with the specified hyperparameters. Network
        weights and biases will be initialized at fitting time following the shape
        of the training data and the specified sizes, which represents the
        amount of units to include in each hidden layer of the current network.
        Each layer will be initialized randomly following the LeCun uniform
        initializer formula. Implements a neural network for classification
        tasks, since it uses the sigmoid activation function in the output
        layer.

        Parameters
        ----------
        sizes : tuple, optional
            Tuple (l1, l2, ..., ln) containig the number of units for each
            layer of the network to be built, by default None

        optimizer : str, optional
            string indicating which optimizer should be used for the training
            of the current network. Must be in {'SGM', 'SGD'}, by default 'SGD'

        seed : int, optional
            seed for random number generator used for initializing this network
            weights and biases. Needed for reproducibility, by default 0
       
        epochs : int, optional
            the maximum number of epochs the training can run until stopping
            if no termination conditions are met, by default 1000
     
        eta : float, optional
            learning rate for 'SGD' optimizer, starting step sizes for 'SGM'
            optimizer, by default 0.1
   
        activation : function, optional
            specifies which activation function to use for the hidden layers
            of the network, must be in {'Lrelu', 'relu', 'sigmoid'},
            by default 'Lrelu'
    
        lmbda : int, optional
            l2 regularization coefficient, by default 0.0001
    
        momentum : int, optional
            momentum coefficient, by default 0.5
    
        nesterov : bool, optional
            boolean flag indicating wether nesterov momentum must be used
            during optimization of the current network, by default False
    
        eps : float, optional
            stopping condition for precision in gradient norm, by default 1e-5
    
        batch_size : int or None, optional
            amount of samples to use for each evaluation of the gradient during
            optimization, by default None
   
        debug : bool, optional
            debugging flag, by default False
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
        # and the specific scoring function
        self.last_act = Sigmoid
        self.loss = mean_squared_error
        self.scoring = accuracy_score
        self.last_pred = None


    def predict(self, data):
        """Performs a feedforward pass through the network for the given :data:
        samples, returns the predicted classification value for each sample.

        Parameters
        ----------
        data : np.ndarray
            samples to use for prediction.

        Returns
        -------
        np.ndarray
            Binary classification prediction values for the given :data: samples.
        """       
        self.last_pred = self._feedforward_batch(data)[2]

        return  self.last_pred >= 0.5



class NR(Network, BaseEstimator):
    def __init__(
        self, sizes=None, optimizer='SGD', seed=0, epochs=1000, eta=0.01,
        activation='Lrelu', lmbda=0.0001, momentum=0.5, nesterov=False, eps=1e-5,
        batch_size=None, debug=False):
        """Initializes the network with the specified hyperparameters. Network
        weights and biases will be initialized at fitting time following the shape
        of the training data and the specified sizes, which represents the
        amount of units to include in each hidden layer of the current network.
        Each layer will be initialized randomly following the LeCun uniform
        initializer formula. Implements a neural network for regression
        tasks, since it uses the linear activation function in the output
        layer.

        Parameters
        ----------
        sizes : tuple, optional
            Tuple (l1, l2, ..., ln) containig the number of units for each
            layer of the network to be built, by default None

        optimizer : str, optional
            string indicating which optimizer should be used for the training
            of the current network. Must be in {'SGM', 'SGD'}, by default 'SGD'

        seed : int, optional
            seed for random number generator used for initializing this network
            weights and biases. Needed for reproducibility, by default 0
       
        epochs : int, optional
            the maximum number of epochs the training can run until stopping
            if no termination conditions are met, by default 1000
     
        eta : float, optional
            learning rate for 'SGD' optimizer, starting step sizes for 'SGM'
            optimizer, by default 0.01
   
        activation : function, optional
            specifies which activation function to use for the hidden layers
            of the network, must be in {'Lrelu', 'relu', 'sigmoid'},
            by default 'Lrelu'
    
        lmbda : int, optional
            l2 regularization coefficient, by default 0.0001
    
        momentum : int, optional
            momentum coefficient, by default 0.5
    
        nesterov : bool, optional
            boolean flag indicating wether nesterov momentum must be used
            during optimization of the current network, by default False
    
        eps : float, optional
            stopping condition for precision in gradient norm, by default 1e-5
    
        batch_size : int or None, optional
            amount of samples to use for each evaluation of the gradient during
            optimization, by default None
   
        debug : bool, optional
            debugging flag, by default False
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
        """Performs a feedforward pass through the network for the given :data:
        samples, returns the values for each sample for the regression task.

        Parameters
        ----------
        data : np.ndarray
            samples to use for prediction.

        Returns
        -------
        np.ndarray
            Regression prediction values for the given :data: samples.
        """
        self.last_pred = self._feedforward_batch(data)[2]
        return self.last_pred