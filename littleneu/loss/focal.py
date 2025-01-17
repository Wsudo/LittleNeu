import numpy

from .loss_function import ImbalancedClassificationLossFunction

from ..errors import LossErrors

class FocalLoss(ImbalancedClassificationLossFunction):
    """Focal Loss is a loss function designed to address the class imbalance problem, commonly used in tasks like object detection (e.g., in RetinaNet). It is an extension of the cross-entropy loss, adding a factor to down-weight the loss for well-classified examples, thus focusing more on hard, misclassified examples.

    Args:
        y_true (numpy.ndarray | list): true values
        y_pred (numpy.ndarray | list): network generated values
        alpha (float) : Weighting factor for balancing class imbalance (default 0.25)
        gamma(int|float): Focusing parameter to down-weight easy examples (default 2)

    Raises:
        LossErrors: when y_true is not type (list , numpy.ndarray)
        LossErrors: when y_pred is not type (list , numpy.ndarray)

    Returns:
        numpy.ndarray: loss between `True Values` and `Predicted Values`
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: int | float = 2,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def calc(
        self,
        y_true: numpy.ndarray | list,
        y_pred: numpy.ndarray | list,
    ) -> numpy.ndarray:
        """Focal Loss is a loss function designed to address the class imbalance problem, commonly used in tasks like object detection (e.g., in RetinaNet). It is an extension of the cross-entropy loss, adding a factor to down-weight the loss for well-classified examples, thus focusing more on hard, misclassified examples.

        Args:
            y_true (numpy.ndarray | list): true values
            y_pred (numpy.ndarray | list): network generated values
            alpha (float) : Weighting factor for balancing class imbalance (default 0.25)
            gamma(int|float): Focusing parameter to down-weight easy examples (default 2)

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

        # Clip predictions to prevent log(0) which would lead to NaN or infinity
        real_y_pred = numpy.clip(real_y_pred, 1e-7, 1 - 1e-7)

        # Calculate p_t: if true class is 1, p_t is real_y_pred; if true class is 0, p_t is (1 - real_y_pred)
        p_t = real_y_true * real_y_pred + (1 - real_y_true) * (1 - real_y_pred)

        # Calculate the focal loss
        loss = -self.alpha * (1 - p_t) ** self.gamma * numpy.log(p_t)

        return numpy.mean(loss)


def focal_loss(
    y_true: numpy.ndarray | list,
    y_pred: numpy.ndarray | list,
    alpha: float = 0.25,
    gamma: int | float = 2,
) -> FocalLoss | numpy.ndarray:
    """Focal Loss is a loss function designed to address the class imbalance problem, commonly used in tasks like object detection (e.g., in RetinaNet). It is an extension of the cross-entropy loss, adding a factor to down-weight the loss for well-classified examples, thus focusing more on hard, misclassified examples.

    Args:
        y_true (numpy.ndarray | list): true values
        y_pred (numpy.ndarray | list): network generated values
        alpha (float) : Weighting factor for balancing class imbalance (default 0.25)
        gamma(int|float): Focusing parameter to down-weight easy examples (default 2)

    Raises:
        LossErrors: when y_true is not type (list , numpy.ndarray)
        LossErrors: when y_pred is not type (list , numpy.ndarray)

    Returns:
        numpy.ndarray: loss between `True Values` and `Predicted Values`
    """
    if y_true is None and y_pred is None:
        return FocalLoss(alpha=alpha, gamma=gamma)

    return FocalLoss(alpha=alpha, gamma=gamma).calc(y_true, y_pred)
