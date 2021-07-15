from abc import ABCMeta, abstractmethod

import numpy as np
from sklearn.metrics import accuracy_score

class LossFunction(metaclass=ABCMeta):

    @abstractmethod
    def loss(y_true, y_pred):
        pass


class MeanSquaredError(LossFunction):
    """Static class the implements the Mean Squared Error loss function.
    """    

    @staticmethod
    def loss(y_true, y_pred):
        """Returns the average squared error loss value given the
        truth values :y_true: and the predicted ones :y_pred:.

        Parameters
        ----------
        y_true : np.ndarray
            Ground truth vector containing the expected values for a given example x_i.
        y_pred : np.ndarray
            Prediction vector containing the predicted values for an example x_i.

        Returns
        -------
        float
            Average squared distortion from the ground truth vector given the predictions.
        """        

        return 1/2 * np.average((np.array(y_true) - np.array(y_pred))**2)

class AccuracyScore(LossFunction):
    """Static class that implements the accuracy score for classifiers.
    """
    # TODO: magari implementarla come si deve a manina
    @staticmethod
    def loss(y_true, y_pred):
        """Returns the accuracy score given the truth values :y_true: and
        the predicted ones :y_pred:.

        Parameters
        ----------
        y_true : np.ndarray
            Ground truth vector containing the expected class for a given example x_i.
        y_pred : [type]
            Prediction vector containing the predicted class for an example x_i.

        Returns
        -------
        float
            Returns the proportion of correctly classified samples among all.
        """        
        return accuracy_score(y_true, y_pred)