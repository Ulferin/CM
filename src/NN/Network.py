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
    """Abstract class representing a standard Neural Network, also known as
    Multilayer Perceptron. It allows to build a network for both classification
    and regression tasks by using the preferred optimization technique between
    sub-gradient method and stochastic gradient descent. Must be extended to
    specify which kind of task the network should solve.
    """

    @abstractmethod
    def __init__(self,
                hidden_layer_sizes=None, solver='sgd', seed=0, max_iter=1000, learning_rate_init=0.01,
                activation='logistic', alpha=0.0, momentum=0.0, nesterovs_momentum=False,
                tol=1e-5, batch_size=None, verbose=False, beta_1=0.9, beta_2=0.999):

        self.rng = default_rng(seed)     # needed for reproducibility

        # Network hyperparameters
        self.batch_size = batch_size
        self.learning_rate_init = learning_rate_init
        self.tol = tol
        self.max_iter = max_iter
        self.solver = solver
        self.seed = seed
        self.verbose = verbose
        self.momentum = momentum
        self.nesterovs_momentum = nesterovs_momentum
        self.alpha = alpha
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.last_act = None       # Must be defined by subclassing the Network

        # Performance attributes
        self.grad_est = []
        self.grad_est_per_epoch = []
        self.val_scores = []
        self.train_scores = []
        self.val_loss = []
        self.train_loss = []
        self.gap = []
        self.f_star = 999
        self.grad_star = 999
        self.grad_gap = []
        self.grad_rate = []

        #self._no_improvement_count = 0
        self.loss_k = -1
        self.loss_k1 = -1
        self.loss_k2 = -1
        self.loss_k3 = -1
        self.conv_rate = []

        # Execution Statistics
        self.evaluate_avg = [0, 0]
        self.backprop_avg = [0, 0]
        self.feedforward_avg = [0, 0]
        self.epochs_time = []
        self.total_time = 0
        self.update_avg = 0

        # Initialize optimizer
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

        # Last layer is linear for regression and logistic for classification
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
        by performing updates using the solver associated to the current
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

            self.opti.update_parameters(params, grads)


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
                        np.hstack([el.ravel() for el in nabla_w + nabla_b]))
        self.grad_est.append(self.ngrad)

        return nabla_b, nabla_w


    def fit(self, X, y, test_data=None, f_star_set=None, grad_star=None):
        """Trains the neural network on (:X:, :y:) samples for a given
        number of max_iter by fine-tuning the weights and biases by using the
        update rules relative to the provided solver. The way updates are
        performed is also determined by the configurations relative to
        batch size and learning_rate_init hyperparameters.

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

        f_star_set : float, optional
            If set the method will also compute the gap = (f_i - f_*) / f_*
            for the model and then plot it , by default None.
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

        # Set up activation function and solver
        self.act = ACTIVATIONS[self.activation]
        self.der = self.act.derivative

        # Set up input/output units
        self._hidden_layer_sizes = self.hidden_layer_sizes.copy()
        self._hidden_layer_sizes.insert(0, self.X.shape[1])
        self._hidden_layer_sizes.append(1 if len(self.y.shape) == 1 else self.y.shape[1])
        self.num_layers = len(self._hidden_layer_sizes)

        # Initialize network parameters
        self.biases = [np.array(self.rng.normal(0,0.5,l)) for l in self._hidden_layer_sizes[1:]]
        self.weights = [
            np.array(self.rng.uniform(-np.sqrt(2/x+y), np.sqrt(2/x+y), (y,x)))
            for x, y in zip(self._hidden_layer_sizes[:-1], self._hidden_layer_sizes[1:])]

        start = dt.now()
        self.fitted = True
        try:
            for e in range(1, self.max_iter+1):
                s = dt.now()
                self._update_batches()
                en = end_time(s)
                self.update_avg += en

                # Compute current gradient estimate
                self.grad_est_per_epoch.append(np.average(self.grad_est))
                if f_star_set:
                    self.evaluate(e, f_star_set, grad_star)
                else:
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



    def evaluate(self, e, f_star_set=None, grad_star=None):
        """Returns statistics for the current epoch if test data are provided
        while training the network. It prints the current epoch, gradient norm
        for convergence analysis and the current score computed as loss value.

        Parameters
        ----------
        e : int
            current epoch.

        f_star_set : float, optional
            If set the method will also compute the gap = (f_i - f_*) / f_*
            for the model and then plot it , by default None.
        """

        start = dt.now()

        self._evaluate((self.X, self.y), self.test_data, f_star_set, grad_star)

        str_gap = ""
        str_rate = ""
        if self.gap:
            str_gap = f"|| gap: {self.gap[-1]:7.5e}"
        if f_star_set is not None and len(self.conv_rate) > 0:
            str_rate = f"|| rate: {self.conv_rate[-1]:7.5e}"

        val_loss = ""
        if self.test_data is not None:
            val_loss = f"{self.val_loss[-1]:7.5e}, "

        if self.verbose:
            print(
                f"{e:<7} || Gradient norm: {self.grad_est_per_epoch[-1]:7.5e} || "
                f"Loss: {val_loss}{self.train_loss[-1]:7.5e} ||"
                f"Score: {self.val_scores[-1]:5.3g}, "
                f"{self.train_scores[-1]:<5.3g} ||"
                f"f_star: {self.f_star:7.5e} "+str_gap+str_rate)

        end = end_time(start)
        self.evaluate_avg[0] += 1
        self.evaluate_avg[1] += end


    def _evaluate(self, train_data, test_data=None, f_star_set=None, grad_star=None):
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

        f_star_set : float, optional
            If set the method will also compute the gap = (f_i - f_*) / f_*
            for the model and then plot it , by default None.

        """
        if test_data:
            preds_test = self.predict(test_data[0])
            truth_test = test_data[1]

            loss_test = self.loss(truth_test, self.last_pred)
            values_test = 0
            for w in self.weights:
                w = w.ravel()
                values_test += np.linalg.norm(w, 2, 0)
            loss_test += 0.5*self.alpha*values_test/self.training_size

            self.val_loss.append(loss_test)
            self.val_scores.append(self.scoring(truth_test, preds_test))

        preds_train = self.predict(train_data[0])
        truth_train = train_data[1]

        loss = self.loss(truth_train, self.last_pred)
        values = 0
        for w in self.weights:
            w = w.ravel()
            values += np.linalg.norm(w, 2, 0)
        loss += (0.5 * self.alpha) * values / self.training_size

        #improvement of at least tol
        if loss < self.f_star:
            self.f_star = loss
        
        if self.grad_est_per_epoch[-1] < self.grad_star:
            self.grad_star = self.grad_est_per_epoch[-1]

        if f_star_set:
            current_gap = np.abs((loss - f_star_set))/np.abs(f_star_set)
            self.gap.append(current_gap)

            grad_gap = np.abs(self.grad_est_per_epoch[-1] - grad_star)/np.abs(grad_star)
            self.grad_gap.append(grad_gap)

        self.train_loss.append(loss)
        self.train_scores.append(self.scoring(truth_train, preds_train))

        self.loss_k = self.loss_k1
        self.loss_k1 = self.loss_k2
        self.loss_k2 = self.loss_k3
        self.loss_k3 = loss
        if f_star_set and len(self.train_loss) > 4:
            abs_err_top = np.abs((self.loss_k3 - f_star_set + 1e-16)/(self.loss_k2 - f_star_set + 1e-16))
            abs_err_bot = np.abs((self.loss_k2 - f_star_set + 1e-16)/(self.loss_k1 - f_star_set + 1e-16))
            # p = np.log(abs_err_top) / np.log(abs_err_bot)
            p = np.log(abs_err_top  + 1e-16) / np.log(abs_err_bot + 1e-16)
            self.conv_rate.append(p)

        if f_star_set and len(self.train_loss) > 1:
            self.grad_rate.append(self.grad_gap[-1]/self.grad_gap[-2])


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
            whether build plots according to number of max_iter or time of
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
            plt.plot(x, val_res, label='Validation loss')
        plt.plot(x, train_res, label='Training loss')

        plt.xlabel(x_label)
        plt.ylabel (curve_type)
        if log: plt.yscale('log')

        plt.legend(loc='best')
        plt.title (f'{curve_type} {self.solver}')
        plt.draw()


        if save:
            # plt.savefig(
            #     f"src/NN/res/stats/{name}"
            #     f"ep{self.max_iter}s{self.hidden_layer_sizes}b{self.batch_size}e{self.learning_rate_init}"
            #     f"alpha{self.alpha}m{self.momentum}.png")
            plt.savefig(
                f"plots/loss_{name}.png")
        else:
            plt.show()
        plt.clf()


    def plot_grad(self, name, save=False, time=False, gap=False):
        """Builds a plot of the gradient values achieved during training of the
        current network.

        Parameters
        ----------
        name : string
            prefix file name for the plot file, only used when save=True.

        save : bool, optional
            whether to save the plot on file or not, by default False

        time : bool, optional
            whether to plot gradient w.r.t. max_iter or execution time,
            by default False
        """

        if not self.fitted:
            return 'This model is not fitted yet.\n\n'

        x = self.epochs_time if time else list(range(len(self.epochs_time)))
        x_label = 'Execution Time' if time else 'Epochs'
        title = ''
        if gap:
            title = 'Gradient norm gap'
            name = name + '_gap'
            plt.plot(x, self.grad_gap, label='')
        else:
            title = 'Gradient norm'
            plt.plot(x, self.grad_est_per_epoch, label='')
#         plt.legend(loc='best')
        plt.xlabel (x_label)
        plt.ylabel ('Gradient\'s norm')
        plt.title (title)
        plt.yscale('log')
        plt.draw()

        if save:
            # plt.savefig(
            #     f"src/NN/res/stats/{name}ep{self.max_iter}s{self.hidden_layer_sizes}"
            #     f"b{self.batch_size}e{self.learning_rate_init}alpha{self.alpha}"
            #     f"m{self.momentum}.png")
            plt.savefig(
                f"plots/grad_{name}.png")
        else:
            plt.show()
        plt.clf()

    def plot_rate(self, name, save=False):
        if not self.fitted:
            return 'This model is not fitted yet.\n\n'

        x = list(range(1, len(self.conv_rate[:-1]) + 1))
        x_label = 'Epochs'

        plt.plot(x, self.conv_rate[:-1], label='rate')
        plt.legend(loc='best')
        plt.xlabel (x_label)
        plt.ylabel ('Convergence rate')
        plt.title ('Convergence rate per epoch')
        plt.yscale('log')
        plt.draw()

        if save:
            plt.savefig(
                f"./plots/rate_{name}.png")
        else:
            plt.show()
        plt.clf()

    def plot_grad_rate(self, name, save=False):
        if not self.fitted:
            return 'This model is not fitted yet.\n\n'

        x = list(range(1, len(self.grad_gap[1:]) + 1))
        x_label = 'Epochs'

        grad_rate = []
        for i in range(len(self.grad_gap[1:])):
            grad_rate.append(self.grad_gap[i]/self.grad_gap[i-1])

        plt.plot(x, grad_rate, label='rate')
        plt.legend(loc='best')
        plt.xlabel (x_label)
        plt.ylabel ('Convergence rate')
        plt.title ('Convergence rate per epoch')
        plt.yscale('log')
        plt.draw()

        if save:
            plt.savefig(
                f"./plots/grad_rate_{name}.png")
        else:
            plt.show()
        plt.clf()


    def plot_gap(self, dataset, solver, save=True):
        epochs = range(len(self.gap))

        plt.plot(epochs, self.gap, label='gap')
        plt.legend(loc='best')
        plt.xlabel ('epochs')
        plt.ylabel ('gap term')
        plt.title ('Gap term '+dataset+' with '+solver)
        plt.yscale('log')
        plt.draw()

        if save:
            plt.savefig(
                f"./plots/gap_{dataset}_{solver}.png")
        else:
            plt.show()
        plt.clf()



class NC(Network, BaseEstimator):
    def __init__(
        self, hidden_layer_sizes=None, solver='sgd', seed=0, max_iter=300, learning_rate_init=0.1,
        activation='logistic', alpha=0.0001, momentum=0.5, nesterovs_momentum=False,
        tol=1e-5, batch_size=None, verbose=False, beta_1=0.9, beta_2=0.999):
        """Initializes the network with the specified hyperparameters. Network
        weights and biases will be initialized at fitting time following the shape
        of the training data and the specified hidden_layer_sizes, which represents the
        amount of units to include in each hidden layer of the current network.
        Each layer will be initialized randomly following the LeCun uniform
        initializer formula. Implements a neural network for classification
        tasks, since it uses the logistic activation function in the output
        layer.

        Parameters
        ----------
        hidden_layer_sizes : tuple, optional
            Tuple (l1, l2, ..., ln) containig the number of units for each
            layer of the network to be built, by default None

        solver : str, optional
            string indicating which solver should be used for the training
            of the current network. Must be in {'SGM', 'sgd'}, by default 'sgd'

        seed : int, optional
            seed for random number generator used for initializing this network
            weights and biases. Needed for reproducibility, by default 0

        max_iter : int, optional
            the maximum number of max_iter the training can run until stopping
            if no termination conditions are met, by default 1000

        learning_rate_init : float, optional
            learning rate for 'sgd' solver, starting step sizes for 'SGM'
            solver, by default 0.1

        activation : function, optional
            specifies which activation function to use for the hidden layers
            of the network, must be in {'Lrelu', 'relu', 'logistic'},
            by default 'Lrelu'

        alpha : int, optional
            l2 regularization coefficient, by default 0.0001

        momentum : int, optional
            momentum coefficient, by default 0.5

        nesterovs_momentum : bool, optional
            boolean flag indicating wether nesterovs_momentum momentum must be used
            during optimization of the current network, by default False

        tol : float, optional
            stopping condition for precision in gradient norm, by default 1e-5

        batch_size : int or None, optional
            amount of samples to use for each evaluation of the gradient during
            optimization, by default None

        verbose : bool, optional
            debugging flag, by default False
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
        self, hidden_layer_sizes=None, solver='sgd', seed=0, max_iter=1000, learning_rate_init=0.01,
        activation='logistic', alpha=0.0001, momentum=0.5, nesterovs_momentum=False, tol=1e-5,
        batch_size=None, verbose=False, beta_1=0.9, beta_2=0.999):
        """Initializes the network with the specified hyperparameters. Network
        weights and biases will be initialized at fitting time following the shape
        of the training data and the specified hidden_layer_sizes, which represents the
        amount of units to include in each hidden layer of the current network.
        Each layer will be initialized randomly following the LeCun uniform
        initializer formula. Implements a neural network for regression
        tasks, since it uses the linear activation function in the output
        layer.

        Parameters
        ----------
        hidden_layer_sizes : tuple, optional
            Tuple (l1, l2, ..., ln) containig the number of units for each
            layer of the network to be built, by default None

        solver : str, optional
            string indicating which solver should be used for the training
            of the current network. Must be in {'SGM', 'sgd'}, by default 'sgd'

        seed : int, optional
            seed for random number generator used for initializing this network
            weights and biases. Needed for reproducibility, by default 0

        max_iter : int, optional
            the maximum number of max_iter the training can run until stopping
            if no termination conditions are met, by default 1000

        learning_rate_init : float, optional
            learning rate for 'sgd' solver, starting step sizes for 'SGM'
            solver, by default 0.01

        activation : function, optional
            specifies which activation function to use for the hidden layers
            of the network, must be in {'Lrelu', 'relu', 'logistic'},
            by default 'Lrelu'

        alpha : int, optional
            l2 regularization coefficient, by default 0.0001

        momentum : int, optional
            momentum coefficient, by default 0.5

        nesterovs_momentum : bool, optional
            boolean flag indicating wether nesterovs_momentum momentum must be used
            during optimization of the current network, by default False

        tol : float, optional
            stopping condition for precision in gradient norm, by default 1e-5

        batch_size : int or None, optional
            amount of samples to use for each evaluation of the gradient during
            optimization, by default None

        verbose : bool, optional
            debugging flag, by default False
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
            samples to use for prediction.

        Returns
        -------
        np.ndarray
            Regression prediction values for the given :data: samples.
        """
        self.last_pred = self._feedforward_batch(data)[2]
        return self.last_pred
