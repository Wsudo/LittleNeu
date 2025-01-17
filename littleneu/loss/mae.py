import numpy

from .loss_function import RegressionLossFunction

from ..errors import LossErrors

class MAELoss(RegressionLossFunction):

    def __init__(self):
        super().__init__()

    def calc(
        self,
        y_true: numpy.ndarray | list,
        y_pred: numpy.ndarray | list,
    ) -> numpy.ndarray:
        """The MAE loss is computed by taking the absolute difference between the true values and the predicted values and then averaging these differences.

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

        abs_diff = numpy.abs(real_y_true, real_y_pred)

        return numpy.mean(abs_diff)


def mae_loss(
    y_true: numpy.ndarray | list = None,
    y_pred: numpy.ndarray | list = None,
) -> MAELoss | numpy.ndarray:
    """The MAE loss is computed by taking the absolute difference between the true values and the predicted values and then averaging these differences.

    Args:
        y_true (numpy.ndarray | list): true values
        y_pred (numpy.ndarray | list): network generated values

    Raises:
        LossErrors: when y_true is not type (list , numpy.ndarray)
        LossErrors: when y_pred is not type (list , numpy.ndarray)

    Returns:
        numpy.ndarray: loss between `True Values` and `Predicted Values`
    """
    return MAELoss() if y_true == None and y_pred == None else MAELoss().calc(y_true, y_pred)
