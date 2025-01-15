import numpy

from .loss_function import ClassificationLossFunction


class KLLoss(ClassificationLossFunction):
    """The KL divergence is computed by iterating through each element in the distributions p and q, and applying the formula P(i)log⁡(P(i))P(i)log(Q(i)P(i)​) for each corresponding pair of elements in p and q.

    Args:
        y_true (numpy.ndarray | list): true values
        y_pred (numpy.ndarray | list): network generated values

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
        """The KL divergence is computed by iterating through each element in the distributions p and q, and applying the formula P(i)log⁡(P(i))P(i)log(Q(i)P(i)​) for each corresponding pair of elements in p and q.

        Args:
            y_true (numpy.ndarray | list): true values
            y_pred (numpy.ndarray | list): network generated values

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

        real_y_true = numpy.clip(real_y_true, 1e-15, 1)  # True distribution
        real_y_pred = numpy.clip(real_y_pred, 1e-15, 1)  # Predicted distribution

        # Calculate KL Divergence: D_KL(P || Q) = sum(P(i) * log(P(i) / Q(i)))
        kl_div = numpy.sum(real_y_true * numpy.log(real_y_true / real_y_pred))

        return kl_div


class KullBackLeiblerLoss(KLLoss):
    """The KL divergence is computed by iterating through each element in the distributions p and q, and applying the formula P(i)log⁡(P(i))P(i)log(Q(i)P(i)​) for each corresponding pair of elements in p and q.

    Args:
        y_true (numpy.ndarray | list): true values
        y_pred (numpy.ndarray | list): network generated values

    Raises:
        ValueError: when y_true is not type (list , numpy.ndarray)
        ValueError: when y_pred is not type (list , numpy.ndarray)

    Returns:
        numpy.ndarray: loss between `True Values` and `Predicted Values`
    """

    def __init__(self):
        super().__init__()


def kl_loss(
    y_true: numpy.ndarray | list = None,
    y_pred: numpy.ndarray | list = None,
) -> KLLoss | numpy.ndarray:
    """The KL divergence is computed by iterating through each element in the distributions p and q, and applying the formula P(i)log⁡(P(i))P(i)log(Q(i)P(i)​) for each corresponding pair of elements in p and q.

    Args:
        y_true (numpy.ndarray | list): true values
        y_pred (numpy.ndarray | list): network generated values

    Raises:
        ValueError: when y_true is not type (list , numpy.ndarray)
        ValueError: when y_pred is not type (list , numpy.ndarray)

    Returns:
        numpy.ndarray: loss between `True Values` and `Predicted Values`
    """
    return KLLoss() if y_true is None and y_pred is None else KLLoss().calc(y_true, y_pred)


def kullback_leibler_loss(
    y_true: numpy.ndarray | list = None,
    y_pred: numpy.ndarray | list = None,
) -> KullBackLeiblerLoss | numpy.ndarray:
    """The KL divergence is computed by iterating through each element in the distributions p and q, and applying the formula P(i)log⁡(P(i))P(i)log(Q(i)P(i)​) for each corresponding pair of elements in p and q.

    Args:
        y_true (numpy.ndarray | list): true values
        y_pred (numpy.ndarray | list): network generated values

    Raises:
        ValueError: when y_true is not type (list , numpy.ndarray)
        ValueError: when y_pred is not type (list , numpy.ndarray)

    Returns:
        numpy.ndarray: loss between `True Values` and `Predicted Values`
    """
    return KullBackLeiblerLoss() if y_true is None and y_pred is None else KullBackLeiblerLoss().calc(y_true, y_pred)
