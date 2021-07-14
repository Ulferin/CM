from abc import ABCMeta, abstractmethod

import numpy as np

from src.NN.LossFunctions import MeanSquaredError

class Optimizer(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self, training_data, epochs, eta, batch_size=None, test_data=None):
        # Store auxiliary informations to pretty-print statistics
        self.batch_size = batch_size
        self.eta = eta
        self.epochs = epochs
        self.training_size = len(training_data[0])
        self.batches = int(self.training_size/batch_size) if batch_size is not None else 1
        self.grad_est_per_epoch = []

        # TODO: magari questo si può mettere nelle specifiche indicando che sia train che test devono avere vettore obiettivo come 2d vector
        # Reshape vectors to fit needed shape
        self.training_data = (training_data[0], training_data[1].reshape(training_data[1].shape[0], -1))
        self.test_data = test_data


    def _create_batches(self, batches, training_data):
        """Creates a list of mini-batches that will be used during optimization.
        Each time a new mini-batch is created, the data are shuffled, it is not
        guaranteed that each bach will have different elements than the other ones.

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


    def _evaluate(self, e, nn):
        """Utility function acting as a wrapper around the evaluate method of the Neural Network :nn:.
        Returns statistics for the current epoch if test data are provided while calling the optimizer.
        It prints the current epoch, gradient norm for convergence analysis and the current score computed
        as loss value.

        Parameters
        ----------
        e : int
            current epoch.
        nn : Object
            Neural Network object used during optimization.
        """              

        if self.test_data is not None:
            score, preds_train, preds_test = nn.evaluate(self.test_data, self.training_data)
            if nn.debug: print(f"pred train: {preds_train[1]} --> target: {self.training_data[1][1]} || pred test: {preds_test[1]} --> target {self.test_data[1][1]}")
            print(f"Epoch {e}. Gradient norm: {self.grad_est}. Score: {score}")
        else:
            print(f"Epoch {e} completed.")


    def _compute_grad(self, nn, mini_batch, der):
        """Computes the gradient values and norm for the current :mini_batch: samples by
        using the backpropagation method implemented by the :nn: Neural Network object.
        The derivative method to be used is specified by :der: indicating whether a gradient
        or subgradient computation should be used.

        Parameters
        ----------
        nn : Object
            Neural Network object used for backpropagation.
        mini_batch : np.ndarray
            mini-batch samples for which to compute the gradient.
        der : function
            Indicated the technique used to compute the gradient.

        Returns
        -------
        np.ndarray, np.ndarray
            Couple of np.ndarray indicating the gradient values for both
            weights and biases.
        """        

        nabla_b, nabla_w = nn._backpropagation_batch(mini_batch[0], mini_batch[1], der)
        self.ngrad = np.linalg.norm(np.hstack([el.ravel() for el in nabla_w + nabla_b])/len(mini_batch[0]))

        return nabla_b, nabla_w


    def _update_batches(self, nn):
        """Creates batches and updates the Neural Network weights and biases by using the
        number of batches for the current optimizer.

        Parameters
        ----------
        nn : Object
            Neural Network object to use for updates.
        """               

        mini_batches = self._create_batches(self.batches, self.training_data)
        self.grad_est = 0

        for mini_batch in mini_batches:
            self._update_mini_batch(nn, mini_batch)
            self.grad_est += self.ngrad
        self.grad_est = self.grad_est/self.batches


    @abstractmethod
    def _update_mini_batch(self, nn, mini_batch):
        """Updates weights and biases of the specified Neural Network object :nn: by
        using the current mini-batch samples :mini_batch:.

        Parameters
        ----------
        nn : Object
            Neural Network to use for updates and gradient computation.
        mini_batch : np.ndarray
            mini-batch samples to use for updates.
        """              
        pass


    @abstractmethod
    def optimize(self, nn):   
        pass


class SGD(Optimizer):

    def __init__(self, training_data, epochs, eta, batch_size=None, test_data=None):
        super().__init__(training_data, epochs, eta, batch_size=batch_size, test_data=test_data)


    def optimize(self, nn):
        """Trains the Neural Network :nn: using mini-batch stochastic gradient descent,
        applied to the training examples for the current optimizer for a given
        number of epochs and with the specified learning rate. If test_data exists,
        the learning algorithm will print progresses during the training phase.

        Parameters
        ----------
        nn : Object
            Neural Network to train with the current optimization method.
        """           

        for e in range(self.epochs):
            self._update_batches(nn)

            # Compute current gradient estimate
            self.grad_est_per_epoch.append(self.grad_est)
            self._evaluate(e, nn)


    def _update_mini_batch(self, nn, mini_batch):
        """Updates weights and biases of the specified Neural Network object :nn: by
        using the current mini-batch samples :mini_batch:. Uses a regularized momentum
        based approach for the weights update. Hyperparameters must be configured directly
        on the :nn: object.

        Parameters
        ----------
        nn : Object
            Neural Network to use for updates and gradient computation.
        mini_batch : np.ndarray
            mini-batch samples to use for updates.
        """
        
        nabla_b, nabla_w = self._compute_grad(nn, mini_batch, nn.act.derivative)

        # Momentum updates
        nn.wvelocities = [nn.momentum * velocity - (self.eta/len(mini_batch[0]))*nw for velocity,nw in zip(nn.wvelocities, nabla_w)]
        nn.bvelocities = [nn.momentum * velocity - (self.eta/len(mini_batch[0]))*nb for velocity,nb in zip(nn.bvelocities, nabla_b)]

        nn.weights = [w + velocity - (nn.lmbda/len(mini_batch[0]) * w) for w,velocity in zip(nn.weights, nn.wvelocities)]
        nn.biases = [b + velocity for b,velocity in zip(nn.biases, nn.bvelocities)]


class SGM(Optimizer):

    def __init__(self, training_data, epochs, eta, eps=1e-5, batch_size=None, test_data=None):
        super().__init__(training_data, epochs, eta, batch_size=batch_size, test_data=test_data)
        self.eps = eps


    def optimize(self, nn):
        """Trains the Neural Network :nn: using mini-batch sub-gradient method,
        applied to the training examples for the current optimizer for a given
        number of epochs and with the specified step-size. If test_data exists,
        the learning algorithm will print progresses during the training phase.

        Parameters
        ----------
        nn : Object
            Neural Network to train with the current optimization method.
        """

        x_ref = []
        f_ref = np.inf

        for e in range(1, self.epochs+1):
            self.step = self.eta * (1 / e)
            
            preds_train = nn._feedforward_batch(self.training_data[0])[2]
            truth_train = self.training_data[1]

            last_f = MeanSquaredError.loss(truth_train, preds_train)

            self._update_batches(nn)

            # found a better value
            if last_f < f_ref:
                f_ref = last_f
                x_ref = (nn.weights.copy(), nn.biases.copy())

            if self.ngrad < self.eps:
                print("Reached desired accuracy.")
                break

            self._evaluate(e, nn)


    def _update_mini_batch(self, nn, mini_batch):
        """Updates weights and biases of the specified Neural Network object :nn: by
        using the current mini-batch samples :mini_batch:. Uses a diminishing step-size
        rule for updates.

        Parameters
        ----------
        nn : Object
            Neural Network to use for updates and gradient computation.
        mini_batch : np.ndarray
            mini-batch samples to use for updates.
        """

        nabla_b, nabla_w = self._compute_grad(nn, mini_batch, nn.act.subgrad)

        # Compute search direction
        d = self.step/self.ngrad

        nn.weights = [w - d*nw for w,nw in zip(nn.weights, nabla_w)]
        nn.biases = [b - d*nb for b,nb in zip(nn.biases, nabla_b)]