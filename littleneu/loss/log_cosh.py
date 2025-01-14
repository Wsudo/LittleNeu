import numpy

from .loss_function import RegressionLossFunction


class LogCoshLoss(RegressionLossFunction):
    """The log-cosh function behaves similarly to the Mean Squared Error (MSE) when the error is small, but it becomes less sensitive to large errors (like Mean Absolute Error (MAE)).
    This makes Log-Cosh Loss less prone to large gradients for outliers compared to MSE while still being differentiable and smooth.

    Args:
        y_true (numpy.ndarray | list): true values
        y_pred (numpy.ndarray | list): network generated values
        delta (float):is a threshold that defines the point where the loss function transitions from quadratic to linear.

    Raises:
        ValueError: when y_true is not type (list , numpy.ndarray)
        ValueError: when y_pred is not type (list , numpy.ndarray)

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
        """The log-cosh function behaves similarly to the Mean Squared Error (MSE) when the error is small, but it becomes less sensitive to large errors (like Mean Absolute Error (MAE)).
        This makes Log-Cosh Loss less prone to large gradients for outliers compared to MSE while still being differentiable and smooth.

        Args:
            y_true (numpy.ndarray | list): true values
            y_pred (numpy.ndarray | list): network generated values
            delta (float):is a threshold that defines the point where the loss function transitions from quadratic to linear.

        Raises:
            ValueError: when y_true is not type (list , numpy.ndarray)
            ValueError: when y_pred is not type (list , numpy.ndarray)

        Returns:
            numpy.ndarray: loss between `True Values` and `Predicted Values`
        """
        if not isinstance(y_true, numpy.ndarray) and not isinstance(y_true, list):
            raise ValueError(f"{type(self)}.calc 'y_true' argument must be type (list , numpy.ndarray) , {type(y_true)} passed !")
        if not isinstance(y_pred, numpy.ndarray) and not isinstance(y_pred, list):
            raise ValueError(f"{type(self)}.calc 'y_pred' argument must be type (list , numpy.ndarray) , {type(y_pred)} passed !")

        real_y_true = y_true if isinstance(y_true, numpy.ndarray) else numpy.array(y_true)
        real_y_pred = y_pred if isinstance(y_pred, numpy.ndarray) else numpy.array(y_pred)

        loss = numpy.log(numpy.cosh(real_y_pred - real_y_true))

        return loss


def logcosh_loss(
    y_true: numpy.ndarray | list = None,
    y_pred: numpy.ndarray | list = None,
) -> LogCoshLoss | numpy.ndarray:
    """The log-cosh function behaves similarly to the Mean Squared Error (MSE) when the error is small, but it becomes less sensitive to large errors (like Mean Absolute Error (MAE)).
    This makes Log-Cosh Loss less prone to large gradients for outliers compared to MSE while still being differentiable and smooth.

    Args:
        y_true (numpy.ndarray | list): true values
        y_pred (numpy.ndarray | list): network generated values
        delta (float):is a threshold that defines the point where the loss function transitions from quadratic to linear.

    Raises:
        ValueError: when y_true is not type (list , numpy.ndarray)
        ValueError: when y_pred is not type (list , numpy.ndarray)

    Returns:
        numpy.ndarray: loss between `True Values` and `Predicted Values`
    """
    return LogCoshLoss() if y_true is None and y_pred is None else LogCoshLoss().calc(y_true, y_pred)
