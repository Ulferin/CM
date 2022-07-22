# This class represents the Network used in the CM project and it is
# implemented from scratch following the advices taken during the course
# of ML

from datetime import datetime as dt
from abc import ABCMeta, abstractmethod
import pickle
from tqdm import tqdm

import numpy as np
from numpy.random import default_rng
from sklearn.base import BaseEstimator
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt

from src.NN.ActivationFunctions import ReLU, Sigmoid, LeakyReLU, Linear
from src.NN.metrics import mean_squared_error, accuracy_score
from src.NN.optimizers import sgd, Adam
from src.utils import end_time


ACTIVATIONS = {
    'relu': ReLU,
    'Lrelu':LeakyReLU,
    'logistic': Sigmoid
}

OPTIMIZERS = {
    'sgd': sgd,
    'adam': Adam
}


class Network(BaseEstimator, metaclass=ABCMeta):
    """Abstract base class representing a standard Neural Network, also known as
    Multilayer Perceptron. Provides an interface for derived networks solving
    classification and regression tasks. It allows to 'plug-in' various
    optimization methods, implemented in 'optimizers.py'. Must be extended to
    specify which kind of task the network should solve.

    Two derived classes are implemented:
        · NC: Neural Network solving classification tasks. Provides classification
              specific functionalities, such as sigmoid activation function in
              the last layer and accuracy score as score metric.
        · NR: Neural Network solving regression tasks. Provides regression
              specific functionalities, such as linear activation function in
              the last layer and r2_score as score metric.
    """

    @abstractmethod
    def __init__(self,
                hidden_layer_sizes=[2,], solver='sgd', seed=0, max_iter=1000,
                learning_rate_init=0.01, activation='logistic', alpha=0.0,
                momentum=0.0, nesterovs_momentum=False, tol=1e-5, batch_size=None,
                beta_1=0.9, beta_2=0.999, verbose=False):
        """Abstract constructor, defines and initializes common hyperparameters
        for all the implemented networks, like network topology and optimization
        algorithm to use during the training phase.

        NOTE: the naming convention for the hyperparameters names is dictated by
        the necessity to make the implemented classes compatible with the model
        selection utilities provided by the sklearn framework.

       Parameters
        ----------
        hidden_layer_sizes : list, optional
            Network topology, defined as list of integers representing the amount
            of units for each of the hidden layer of the network. Note that the
            input layer and output layer will be automatically determined,
            respectively, by the shape of the training samples and the shape of
            the ground truth vector. By default [2,].

        solver : string, optional
            Optimizer to use to update the network parameters during the training
            phase. Allowed values are {'sgd', 'adam'}. Note that some of the
            network hyperparameters will be meaningful only for specific choices
            of the :solver:, like :momentum: for 'sgd' and :beta_1:/:beta_2: for
            'adam'. By default 'sgd'

        seed : int, optional
            Seed for random number generator, needed for reproducibilty of the
            results, by default 0

        max_iter : int, optional
            Maximum number of iteration the learning process can run. Used as an
            early stopping condition in case the tolerance over the gradient norm
            is not reached, by default 1000

        learning_rate_init : float, optional
            Learning rate for the optimization algorithm, specifies the fixed
            learning rate for the 'sgd' optimizer and the base learning rate for
            the 'adam' optimizer, by default 0.01

        activation : string, optional
            Activation function to use in each of the units of the network,
            except for the output units, which activation function is specific
            to the kind of task the network is going to solve. Available
            activation functions are {'relu', 'Lrelu', 'logistic'}, which
            description is provided in 'ActivationFunctions.py'.
            By default 'logistic'

        alpha : float, optional
            Regularization coefficient, by default 0.0

        momentum : float, optional
            Momentum coefficient, only used when for 'sgd' solver. Should be in
            [0,1), by default 0.0

        nesterovs_momentum : bool, optional
            Whether to use Nesterov's momentum update, only used for 'sgd' solver,
            by default False

        tol : float, optional
            Tolerance on the gradient norm, allows to stop the training process
            when the gradient norm goes below this treshold, by default 1e-5

        batch_size : int, optional
            Number of samples to use for each update of the network parameters,
            must be in [1, max] where max is maximum number of samples in the
            training set. The special value 'None' automatically selects the max
            amount of samples in the provided training set. By default None

        verbose : bool, optional
            Debug flag to make the network print stats during the training process,
            such as the current epoch and the values of the loss and scoring metric
            at the current epoch, by default False

        beta_1 : float, optional
            Exponential decay rate for first moment vector, only used for the
            'adam' solver. Accepted values range in [0,1), by default 0.9

        beta_2 : float, optional
            Exponential decay rate for the second moment vector, only used for
            'adam' solver. Accepted values range in [0,1), by default 0.999
        """

        self.rng = default_rng(seed)     # needed for reproducibility

        # Network hyperparameters
        self.batch_size = batch_size
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.solver = solver
        self.momentum = momentum
        self.nesterovs_momentum = nesterovs_momentum
        self.alpha = alpha
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.last_act = None       # Must be defined by subclassing the Network
        self.verbose = verbose
        self.seed = seed
        self.tol = tol

        # Performance attributes
        self.grad_est = []
        self.grad_est_per_epoch = []
        self.val_scores = []
        self.train_scores = []
        self.val_loss = []
        self.train_loss = []
        self.gap = []
        self.f_star = 999

        # Execution Statistics
        self.evaluate_avg = [0, 0]
        self.backprop_avg = [0, 0]
        self.feedforward_avg = [0, 0]
        self.epochs_time = []
        self.total_time = 0
        self.update_avg = 0

        # Optimizer specific initialization
        optimizer_params = {
            'learning_rate_init': learning_rate_init,
            'tol': tol
        }
        if solver == 'sgd':
            optimizer_params['momentum'] = momentum
            optimizer_params['nesterovs_momentum'] = nesterovs_momentum
        elif solver == 'adam':
            optimizer_params['beta_1'] = beta_1
            optimizer_params['beta_2'] = beta_2

        self.opti = OPTIMIZERS[self.solver](**optimizer_params)


    def _feedforward_batch(self, inp):
        """Performs a foward pass through the network for :inp: input data.
        Returns the network's output for the given input, as well as the 'net'
        function and output of each units in the network.

        Parameters
        ----------
        inp : np.ndarray
            Network input used to perform the forward pass through the network.

        Returns
        -------
        units_out : np.ndarray
            Output computed by the neural network for the provided input data
            :inp:.

        nets : list
            list of unit's activation function input for each layer of the network.
            The input of the activation function is computed as the product
            between the previous layer's output and the weights of the current
            layer, summed the bias term.

        out : list
            List of unit's output for each layer of the network. The values are
            computed as the activation function of the unit's 'net' term.
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

        # Last layer is linear for regression and logistic for classification
        net = np.matmul(out,self.weights[-1].T) + self.biases[-1]
        out = self.last_act.function(net)
        nets.append(net)
        units_out.append(out)

        # Save stats for performance analysis
        end = end_time(start)
        self.feedforward_avg[0] += 1
        self.feedforward_avg[1] += end

        return units_out, nets, out


    def _backpropagation_batch(self, x, y):
        """Performs a backward pass by using the chain rule of calculus to 
        compute the gradient for each weight and bias in the network for the
        current input/output samples.

        Implements the backpropagation algorithm to compute the gradients of
        the objective function which is being optimized by the current network.
        Follows the implementation provided in
        [Thomas M. Mitchell. Machine Learning. 1997.]

        Parameters
        ----------
        x : np.ndarray
            Data samples, used to compute the network's output.

        y : np.ndarray
            Expected output for the :x: samples. Used for determining the amount
            of error for the given samples and to compute the gradients.

        Returns
        -------
        nabla_b : list
            List of np.ndarray containing for each layer the gradients with
            respect to each bias in the network.

        nabla_w : list
            List of np.ndarray containing for each layer the gradients with
            respect to each weight in the network.
        """
        delta = 0
        size = len(x)
        nabla_b = [0]*(len(self._hidden_layer_sizes)-1)
        nabla_w = [0]*(len(self._hidden_layer_sizes)-1)

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

            # Applying regularization and mean above samples
            nabla_b[-l] = delta.sum(axis=0)
            nabla_b[-l] /= size
            nabla_w[-l] = np.matmul(delta.T, units_out[-l-1])
            nabla_w[-l] += np.sign(self.weights[-l])*self.alpha
            nabla_w[-l] /= size

        # Computes execution statistics
        end = end_time(start)
        self.backprop_avg[0] += 1
        self.backprop_avg[1] += end

        return nabla_b, nabla_w


    def _create_batches(self):
        """Utility function used to create batches of data for the training
        process, based on the hyperparameter :batch_size:.

        Returns
        -------
        mini_batches : list
            A list of mini-batches of size :batch_size:, note that the last
            batch may contain less than :batch_size: samples.
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
        """Utility function used to update the network's weights and biases
        using the configured optimization algorithm. Computes the gradients
        of the network function based on the inputs in each of the mini-batches
        and forwards both the parameters and the gradients to the optimizer
        in order to apply the specific update rule for the chosen :solver:.

        This method is called once for each epoch of the training process and
        performs updates using all the data in the training dataset using the
        provided batch configuration (either mini-batch or full-batch).
        """
        mini_batches = self._create_batches()
        self.grad_est = 0
        self.num_batches = len(mini_batches)

        for mini_batch in mini_batches:
            params = self.weights + self.biases

            # Compute current gradient
            nabla_b, nabla_w = self._compute_grad(mini_batch)
            grads = nabla_w + nabla_b

            # Update parameters with solver-specific update rule
            self.opti.update_parameters(params, grads)


    def _compute_grad(self, mini_batch):
        """Computes the gradient values and norm for the current :mini_batch:
        samples by using the backpropagation algorithm. It also keeps track of
        the gradients norm for the current epoch, which will later be used to
        compute the average gradient norm for the current epoch.

        Parameters
        ----------
        mini_batch : np.ndarray
            Data samples used to compute the gradient of the network function.

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
                        np.hstack([el.ravel() for el in nabla_w + nabla_b]))
        self.grad_est += self.ngrad

        return nabla_b, nabla_w


    def fit(self, X, y, test_data=None, f_star_set=None):
        """Trains the neural network on (:X:, :y:) samples for a given
        number of iterations by fine-tuning the weights and biases using the
        update rules relative to the provided solver. The network's weights
        are initialized following the Glorot initialization scheme. 

        Parameters
        ----------
        X : np.ndarray
            Training samples to use to train the neural network. The shape of
            the samples is used to determine the number of input units in the
            first layer of the network.

        y : np.ndarray
            Expected outputs for the given training samples :X:. The shape of the
            ground truth is used to determine the number of output units in the
            last layer of the network.

        test_data : tuple, optional
            If provided, test samples and expected outputs are used to evaluate
            the instantaneous generalization performance of the current network
            at each epoch of training, by default None.

        f_star_set : float, optional
            If set the method will also compute the gap = (f_i - f_*) / f_*
            for the model at each training epoch, by default None.
        
        Returns
        -------
        self : Object
            Returns the fitted network.
        """

        # Initialize batches
        self.X = X
        self.y = y
        self.test_data = test_data
        self.training_size = len(self.X)
        self.batch_size = (
            self.batch_size
            if self.batch_size is not None and self.batch_size > 0
            else self.training_size)
        self.batches = int(self.training_size/self.batch_size)

        # Set up activation function
        self.act = ACTIVATIONS[self.activation]
        self.der = self.act.derivative

        # Set up input/output units
        self._hidden_layer_sizes = self.hidden_layer_sizes.copy()
        self._hidden_layer_sizes.insert(0, self.X.shape[1])
        self._hidden_layer_sizes.append(1 if len(self.y.shape) == 1 else self.y.shape[1])
        self.num_layers = len(self._hidden_layer_sizes)

        # Initialize network parameters
        self.biases = [
            np.array(self.rng.normal(0,0.5,l))
            for l in self._hidden_layer_sizes[1:]
        ]
        self.weights = [
            np.array(self.rng.uniform(-np.sqrt(2/x+y), np.sqrt(2/x+y), (y,x)))
            for x, y in
            zip(self._hidden_layer_sizes[:-1], self._hidden_layer_sizes[1:])
        ]
        
        start = dt.now()
        self.fitted = True
        # Training loop
        try:
            for e in tqdm(range(1, self.max_iter+1)):

                s = dt.now()
                self._update_batches()
                en = end_time(s)
                self.update_avg += en

                # Compute current gradient estimate
                self.grad_est_per_epoch.append(self.grad_est / self.num_batches)
                self.evaluate(e, f_star_set)

                # Check if we reached the desired gradient tolerance
                iteration_end = self.opti.iteration_end(self.ngrad)

                epoch_time = end_time(start)
                self.epochs_time.append(epoch_time)

                if iteration_end:
                    print("Reached desired precision in gradient norm,stopping.")
                    break
        except Exception as VE:
            print(VE)
            self.fitted = False
        finally:
            end = end_time(start)
            self.total_time = end

            if not f_star_set:
                best = np.min(self.train_loss)
                self.gap = [(curr - best) / best for curr in self.train_loss]
            return self



    def evaluate(self, e, f_star_set=None):
        """Returns statistics for the current epoch in terms of training and
        validation objective functions (including the regularization term).
        Additionally, it prints the gradient's norm for the current epoch and the
        gap term if the :f_star_set: parameter is set.

        Parameters
        ----------
        e : int
            Current epoch.

        f_star_set : float, optional
            Best estimated value for the objective function optimized by this
            network, if set the method will also compute the
            gap = (f_i - f_*) / f_* for the current model, by default None.
        """

        start = dt.now()

        self._evaluate((self.X, self.y), self.test_data, f_star_set)

        str_gap = ""
        if self.gap:
            str_gap = f"|| gap: {self.gap[-1]:7.5e}"

        val_loss = ""
        if self.test_data is not None:
            val_loss = f"{self.val_loss[-1]:7.5e}, "

        if self.verbose:
            print(
                f"{e:<7} || Gradient norm: {self.grad_est_per_epoch[-1]:7.5e} || "
                f"Loss: {val_loss}{self.train_loss[-1]:7.5e} ||"
                f"Score: {self.val_scores[-1]:5.3g}, "
                f"{self.train_scores[-1]:<5.3g} ||"
                f"f_star: {self.f_star:7.5e} "+str_gap)

        end = end_time(start)
        self.evaluate_avg[0] += 1
        self.evaluate_avg[1] += end


    def _evaluate(self, train_data, test_data=None, f_star_set=None):
        """Evaluates the performances of the Network in the current state,
        propagating the test and training examples through the network via a
        complete feedforward step. It evaluates the performance using the
        associated loss and regularization term for this Network.

        Parameters
        ----------
        train_data : tuple
            Couple of np.ndarray representing training samples and associated
            outputs. Used to compute performance on training samples.

        test_data : tuple
            Couple of np.ndarray representing test samples and associated
            outputs. Used to test the performances of the network at the current
            phase of the learning process.

        f_star_set : float, optional
            Best estimated value for the objective function optimized by this
            network, if set the method will also compute the
            gap = (f_i - f_*) / f_* for the current model, by default None.
        """
        if test_data:
            preds_test = self.predict(test_data[0])
            truth_test = test_data[1]

            loss_test = self.loss(truth_test, self.last_pred)
            values_test = np.sum([np.sum(np.abs(w)) for w in self.weights])
            loss_test += 0.5*self.alpha*values_test/self.training_size

            self.val_loss.append(loss_test)
            self.val_scores.append(self.scoring(truth_test, preds_test))

        preds_train = self.predict(train_data[0])
        truth_train = train_data[1]

        loss = self.loss(truth_train, self.last_pred)
        values = np.sum([np.sum(np.abs(w)) for w in self.weights])
        loss += (0.5 * self.alpha) * values / self.training_size

        self.train_loss.append(loss)
        self.train_scores.append(self.scoring(truth_train, preds_train))

        if loss < self.f_star:
            self.f_star = loss

        if f_star_set:
            current_gap = np.abs((loss - f_star_set))/np.abs(f_star_set)
            self.gap.append(current_gap)


    def best_score(self):
        """Returns performance statistics recorded during the training process.
        Prints a table containing time statistics as well as performance metrics
        achieved on the training and test sets (if provided upon fitting the 
        network). The table is printed to the console.

        Returns
        -------
        stats : string
            Statistics computed during the network training.
        """

        if not self.fitted:
            return 'This model is not fitted yet.\n\n'

        # FIXME: è il modo giusto di calcolare il best loss???
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
            f"ep: {self.max_iter:<7} | s: {self.hidden_layer_sizes} | b: {self.batch_size} | "
            f"e:{self.learning_rate_init:5} | alpha:{self.alpha:5} | m:{self.momentum:5} | "
            f"nesterovs_momentum: {self.nesterovs_momentum}\n"

            f"Grad: {self.grad_est_per_epoch[-1]:7.5e} | "
            f"Loss: {best_loss[0]:7.5e}, {best_loss[1]:7.5e} | "
            f"Score: {best_score[0]:5.3g}, {best_score[1]:<5.3g}\n"

            f"ended in: {self.total_time}, "
            f"avg per ep: {self.total_time/self.max_iter}\n"

            f"total update: {self.update_avg}, "
            f"avg updt: {self.update_avg/(self.max_iter*self.batches)}\n"

            f"total ff: {self.feedforward_avg[0]}, "
            f"total ff time: {self.feedforward_avg[1]}, "
            f"avg ff: {self.feedforward_avg[1]/self.feedforward_avg[0]}\n"

            f"total bp: {self.backprop_avg[0]}, "
            f"total bp time: {self.backprop_avg[1]}, "
            f"avg bp: {self.backprop_avg[1]/self.backprop_avg[0]}\n"

            f"total ev: {self.evaluate_avg[0]}, "
            f"total ev time: {self.evaluate_avg[1]}, "
            f"avg ev: {self.evaluate_avg[1]/self.evaluate_avg[0]}\n\n")

        return stats


    @abstractmethod
    def predict(self, data):
        """Computes the prediction of the network for the given data. This must
        be implemented by each subclass for the specific task, i.e. classification
        or regression.

        Parameters
        ----------
        data : np.ndarray
            Sample for which to compute the prediction.
        """        
        pass


    def saveModel(self, name):
        """Saves the model in a file.

        Parameters
        ----------
        name : string
            Name of the file to save the model.
        """
        file_path = f"./res/models/{name}.pkl"
        with open(file_path, 'wb+') as f:
            pickle.dump(self, f)


    def loadModel(self, name):
        """Loads the model from a file.

        Parameters
        ----------
        name : string
            Name of the file to load the model.
        """
        file_path = f"./res/models/{name}.pkl"
        with open(file_path, 'rb') as f:
            self = pickle.load(f)
        return self


    def plot_results(self, name, score=False, save=False, time=False, log=False):
        """Builds a plot of the scores achieved during training for the
        validation set and the training set.

        Parameters
        ----------
        name : string
            Prefix file name to use when saving the plot.

        score : bool, optional
            Whether to use the score instead of the objective function,
            by default False

        save : bool, optional
            Whether to save the plot on the file specified with the name
            parameter, by default False

        time : bool, optional
            Whether build plots according to number of max_iter or time of
            execution, by default False

        log : bool, optional
            Whether to use a logplot or not, by default False
        """

        if not self.fitted:
            return 'This model is not fitted yet.\n\n'

        # Conditional configuration
        x_label = 'Execution Time' if time else 'Epochs'
        curve_type = 'Objective function' if not score else 'Score'

        if self.test_data is not None:
            val_res = self.val_scores if score else self.val_loss
        train_res = self.train_scores if score else self.train_loss
        x = self.epochs_time if time else list(range(len(train_res)))

        if self.test_data is not None:
            plt.plot(x, val_res, label='Validation')
        plt.plot(x, train_res, label='Training')

        plt.xlabel(x_label)
        plt.ylabel (curve_type)
        if log: plt.yscale('log')

        plt.legend(loc='best')
        plt.title (f'{curve_type} {self.solver}')
        plt.draw()


        if save:
            plt.savefig(f"tests/NN/plots/loss_{name}.png")
        else:
            plt.show()
        plt.clf()


    def plot_grad(self, name, save=False, time=False):
        """Builds a plot of the gradient norm values achieved during training
        of the current network.

        Parameters
        ----------
        name : string
            Prefix file name for the plot file, only used when save=True.

        save : bool, optional
            Whether to save the plot on file or not, by default False

        time : bool, optional
            Whether to plot gradient norm w.r.t. max_iter or execution time,
            by default False
        """

        if not self.fitted:
            return 'This model is not fitted yet.\n\n'

        x = self.epochs_time if time else list(range(len(self.epochs_time)))
        x_label = 'Execution Time' if time else 'Epochs'
        title = f'Gradient norm over {x_label}'
        plt.plot(x, self.grad_est_per_epoch)
        plt.xlabel (x_label)
        plt.ylabel ('Gradient\'s norm')
        plt.title (title)
        plt.yscale('log')
        plt.draw()

        if save:
            plt.savefig(f"tests/NN/plots/grad_{name}.png")
        else:
            plt.show()
        plt.clf()


    def plot_gap(self, dataset, solver, save=True):
        """Plots the gap term for the objective function of the current
        network over each epoch of the training process.

        Parameters
        ----------
        dataset : string
            Name of the dataset over which the network was trained.
        solver : string
            Solver used to train the network.
        save : bool, optional
            Whether to save the plot to a file, by default True
        """        

        plt.plot(self.gap, label='gap')
        plt.legend(loc='best')
        plt.xlabel ('Epochs')
        plt.ylabel ('Gap term')
        plt.title ('Gap term '+dataset+' with '+solver)
        plt.yscale('log')
        plt.draw()

        if save:
            plt.savefig(
                f"tests/NN/plots/gap_{dataset}_{solver}.png")
        else:
            plt.show()
        plt.clf()



class NC(Network, BaseEstimator):
    def __init__(
        self, hidden_layer_sizes=[2,], solver='sgd', seed=0, max_iter=300, learning_rate_init=0.1,
        activation='logistic', alpha=0.0001, momentum=0.5, nesterovs_momentum=False,
        tol=1e-5, batch_size=None, beta_1=0.9, beta_2=0.999, verbose=False):
        """Initializes the network with the specified hyperparameters. Network
        weights and biases will be initialized at fitting time following the shape
        of the training data and the specified hidden_layer_sizes, which
        represents the amount of units to include in each hidden layer of the
        current network. Each layer will be initialized randomly following Glorot
        uniform initializer formula. Implements a neural network for classification
        tasks, since it uses the logistic activation function in the output
        layer.

        For description of the hyperparameters, see the documentation of the
        Network base class.
        """

        super().__init__(hidden_layer_sizes=hidden_layer_sizes,
                        solver=solver,
                        seed=seed,
                        max_iter=max_iter,
                        learning_rate_init=learning_rate_init,
                        activation=activation,
                        alpha=alpha,
                        momentum=momentum,
                        nesterovs_momentum=nesterovs_momentum,
                        verbose=verbose,
                        tol=tol,
                        batch_size=batch_size,
                        beta_1=beta_1,
                        beta_2=beta_2)

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
            Samples to use for prediction.

        Returns
        -------
        np.ndarray
            Binary classification prediction values for the given :data: samples.
        """
        self.last_pred = self._feedforward_batch(data)[2]

        return  self.last_pred >= 0.5



class NR(Network, BaseEstimator):
    def __init__(
        self, hidden_layer_sizes=[2,], solver='sgd', seed=0, max_iter=1000, learning_rate_init=0.01,
        activation='logistic', alpha=0.0001, momentum=0.5, nesterovs_momentum=False, tol=1e-5,
        batch_size=None, beta_1=0.9, beta_2=0.999, verbose=False):
        """Initializes the network with the specified hyperparameters. Network
        weights and biases will be initialized at fitting time following the shape
        of the training data and the specified hidden_layer_sizes, which
        represents the amount of units to include in each hidden layer of the
        current network. Each layer will be initialized randomly following Glorot
        uniform initializer formula. Implements a neural network for regression
        tasks, since it uses the linear activation function in the output
        layer.

        For description of the hyperparameters, see the documentation of the
        Network base class.
        """
        super().__init__(hidden_layer_sizes=hidden_layer_sizes,
                        solver=solver,
                        seed=seed,
                        max_iter=max_iter,
                        learning_rate_init=learning_rate_init,
                        activation=activation,
                        alpha=alpha,
                        momentum=momentum,
                        nesterovs_momentum=nesterovs_momentum,
                        verbose=verbose,
                        tol=tol,
                        batch_size=batch_size,
                        beta_1=beta_1,
                        beta_2=beta_2)


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
            Samples to use for prediction.

        Returns
        -------
        np.ndarray
            Regression prediction values for the given :data: samples.
        """

        self.last_pred = self._feedforward_batch(data)[2]
        return self.last_pred
