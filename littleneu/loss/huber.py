import numpy

from .loss_function import RegressionLossFunction


class HuberLoss(RegressionLossFunction):
    """Huber Loss is a combination of Mean Squared Error (MSE) and Mean Absolute Error (MAE), which is less sensitive to outliers in data compared to MSE.

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

    def __init__(self, delta: float = 1.0):
        super().__init__()

        self.delta = delta

    def calc(
        self,
        y_true: numpy.ndarray | list,
        y_pred: numpy.ndarray | list,
    ) -> numpy.ndarray:
        """Huber Loss is a combination of Mean Squared Error (MSE) and Mean Absolute Error (MAE), which is less sensitive to outliers in data compared to MSE.

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

        error = numpy.abs(real_y_true - real_y_pred)
        loss = numpy.where(error <= self.delta, 0.5 * (error**2), self.delta * (error - 0.5 * self.delta))
        return numpy.mean(error)


def huber_loss(
    y_true: numpy.ndarray | list = None,
    y_pred: numpy.ndarray | list = None,
    delta: float = 1.0,
) -> HuberLoss | numpy.ndarray:
    """Huber Loss is a combination of Mean Squared Error (MSE) and Mean Absolute Error (MAE), which is less sensitive to outliers in data compared to MSE.

    Args:
        y_true (numpy.ndarray | list): true values
        y_pred (numpy.ndarray | list): network generated values
        delta (float):is a threshold that defines the point where the loss function transitions from quadratic to linear.

    Raises:
        ValueError: when y_true is not type (list , numpy.ndarray)
        ValueError: when y_pred is not type (list , numpy.ndarray)

    Returns:
        numpy.ndarray: loss between `True Values` and `Predicted Values`
        HuberLoss: loss between `True Values` and `Predicted Values`
    """
    return HuberLoss(delta=delta) if y_true is None and y_pred is None else HuberLoss(delta=delta).calc(y_true, y_pred)
