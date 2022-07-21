import numpy as np


def mean_squared_error(y_true, y_pred):
    """Returns the average squared error value given the
    truth values :y_true: and the predicted ones :y_pred:.
    Used as loss function in the implemented neural network.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth vector containing the expected values for a give set of
        samples x.
    y_pred : np.ndarray
        Prediction vector containing the predicted values for the set of samples
        x.

    Returns
    -------
        : float
        Average squared distortion from the ground truth vector given the predictions.
    """        

    return 1/2 * np.average((np.array(y_true) - np.array(y_pred))**2)


def accuracy_score(y_true, y_pred):
    """Returns the accuracy score given the truth values :y_true: and
    the predicted ones :y_pred:.
    Used as scoring metric for classification tasks, returns the percentage of
    correct predictions.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth vector containing the expected class for a set of samples x.
    y_pred : [type]
        Prediction vector containing the predicted class for the set of samples x.

    Returns
    -------
        :float
        Returns the proportion of correctly classified samples.
    """       

    return np.average(y_true == y_pred)