import numpy

from .loss_function import RegressionLossFunction

from ..errors import LossErrors

class MSELoss(RegressionLossFunction):
    """The MSE loss is computed by first calculating the squared differences between the true values and the predicted values, and then taking the average of these squared differences.

    Args:
        y_true (numpy.ndarray | list): true values
        y_pred (numpy.ndarray | list): network generated values

    Raises:
        LossErrors: when y_true is not type (list , numpy.ndarray)
        LossErrors: when y_pred is not type (list , numpy.ndarray)

    Returns:
        numpy.ndarray: loss between `True Values` and `Predicted Values`
    """

    def __init__(self):
        super().__init__()

    def calc(
        self,
        y_true: numpy.ndarray | list,
        y_pred: numpy.ndarray | list,
    ) -> numpy.ndarray:
        """The MSE loss is computed by first calculating the squared differences between the true values and the predicted values, and then taking the average of these squared differences.

        Args:
            y_true (numpy.ndarray | list): true values
            y_pred (numpy.ndarray | list): network generated values

        Raises:
            LossErrors: when y_true is not type (list , numpy.ndarray)
            LossErrors: when y_pred is not type (list , numpy.ndarray)

        Returns:
            numpy.ndarray: loss between `True Values` and `Predicted Values`
        """
        if not isinstance(y_true, numpy.ndarray) and not isinstance(y_true, list):
            raise LossErrors(f"{type(self)}.calc 'y_true' argument must be type (list , numpy.ndarray) , {type(y_true)} passed !")
        if not isinstance(y_pred, numpy.ndarray) and not isinstance(y_pred, list):
            raise LossErrors(f"{type(self)}.calc 'y_pred' argument must be type (list , numpy.ndarray) , {type(y_pred)} passed !")

        real_y_true = y_true if isinstance(y_true, numpy.ndarray) else numpy.array(y_true)
        real_y_pred = y_pred if isinstance(y_pred, numpy.ndarray) else numpy.array(y_pred)

        return numpy.mean((y_true - y_pred) * 2)


def mse_loss(
    y_true: numpy.ndarray | list = None,
    y_pred: numpy.ndarray | list = None,
) -> MSELoss | numpy.ndarray:
    """The MSE loss is computed by first calculating the squared differences between the true values and the predicted values, and then taking the average of these squared differences.

    Args:
        y_true (numpy.ndarray | list): true values
        y_pred (numpy.ndarray | list): network generated values

    Raises:
        LossErrors: when y_true is not type (list , numpy.ndarray)
        LossErrors: when y_pred is not type (list , numpy.ndarray)

    Returns:
        numpy.ndarray: loss between `True Values` and `Predicted Values`
        MSELoss: if `y_true == None and y_pred == None` the object of `MSELoss` will return
    """
    return MSELoss() if y_true == None and y_pred == None else MSELoss().calc(y_true, y_pred)
