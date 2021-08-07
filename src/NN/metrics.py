import numpy as np


def mean_squared_error(y_true, y_pred):
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


def accuracy_score(y_true, y_pred):
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
        Returns the proportion of correctly classified samples.
    """       

    return np.average(y_true == y_pred)