from abc import ABCMeta, abstractmethod

import numpy as np

class LossFunction(metaclass=ABCMeta):

    @abstractmethod
    def loss(y_true, y_pred):
        pass


class MeanSquaredError(LossFunction):

    @staticmethod
    def loss(y_true, y_pred):
        return 1/2 * np.average((np.array(y_true) - np.array(y_pred))**2)