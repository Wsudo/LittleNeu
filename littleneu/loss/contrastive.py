import numpy

from .loss_function import MetricLearningLossFunction


class ContrastiveLoss(MetricLearningLossFunction):
    """Contrastive Loss is commonly used in metric learning tasks to learn an embedding space where similar pairs of examples are closer, and dissimilar pairs are farther apart. It's typically used for Siamese Networks or triplet-based architectures.

    Args:
        y_true (numpy.ndarray|list): True Values
        y_pred (numpy.ndarray|list): Predicted Values from network
        margin (float):Margin for dissimilar pairs (default 1.0)

    Raises:
        ValueError: when y_true is not type (list , numpy.ndarray)
        ValueError: when y_pred is not type (list , numpy.ndarray)
        ValueError: when margin is not type (float)

    Returns:
        numpy.ndarray: the Contrastive loss value (scalar).
    """

    def __init__(
        self,
        margin: float = 0.2,
    ):
        super().__init__()

        self.margin = margin

    def euclidean_distance(
        self,
        x: numpy.ndarray | list,
        y: numpy.ndarray | list,
    ) -> numpy.ndarray:
        """Compute the Euclidean distance between two vectors x and y.

        Args:
            x (numpy.ndarray | list): (anchor) Embedding of the anchor sample (numpy array)
            y (numpy.ndarray | list): (positive/negetive) Embedding of the positive/negetive sample (numpy array)

        Raises:
            ValueError: when x is not type(list,numpy.ndarray)
            ValueError: when y is not type(list,numpy.ndarray)

        Returns:
            numpy.ndarray: output of `Euclidean distance (x , y)`
        """

        if not isinstance(x, numpy.ndarray) and not isinstance(x, list):
            raise ValueError(f"{type(self)}.calc 'x' argument must be type (list , numpy.ndarray) , {type(x)} passed !")
        if not isinstance(y, numpy.ndarray) and not isinstance(y, list):
            raise ValueError(f"{type(self)}.calc 'y' argument must be type (list , numpy.ndarray) , {type(y)} passed !")

        real_x = x if isinstance(x, numpy.ndarray) else numpy.array(x)
        real_y = y if isinstance(y, numpy.ndarray) else numpy.array(y)

        return numpy.linalg.norm(real_x - real_y)

    def calc(
        self,
        y_true: numpy.ndarray | list,
        y_pred: numpy.ndarray | list,
    ) -> numpy.ndarray:
        """Contrastive Loss is commonly used in metric learning tasks to learn an embedding space where similar pairs of examples are closer, and dissimilar pairs are farther apart. It's typically used for Siamese Networks or triplet-based architectures.

        Args:
            y_true (numpy.ndarray|list): True Values
            y_pred (numpy.ndarray|list): Predicted Values from network
            margin (float):Margin for dissimilar pairs (default 1.0)

        Raises:
            ValueError: when y_true is not type (list , numpy.ndarray)
            ValueError: when y_pred is not type (list , numpy.ndarray)
            ValueError: when margin is not type (float)

        Returns:
            numpy.ndarray: the Contrastive loss value (scalar).
        """
        if not isinstance(y_true, numpy.ndarray) and not isinstance(y_true, list):
            raise ValueError(f"{type(self)}.calc 'y_true' argument must be type (list , numpy.ndarray) , {type(y_true)} passed !")
        if not isinstance(y_pred, numpy.ndarray) and not isinstance(y_pred, list):
            raise ValueError(f"{type(self)}.calc 'y_pred' argument must be type (list , numpy.ndarray) , {type(y_pred)} passed !")

        real_y_true = y_true if isinstance(y_true, numpy.ndarray) else numpy.array(y_true)
        real_y_pred = y_pred if isinstance(y_pred, numpy.ndarray) else numpy.array(y_pred)

        # Calculate the contrastive loss
        loss = 0.5 * (real_y_true * numpy.square(real_y_pred) + (1 - real_y_true) * numpy.square(numpy.maximum(0, self.margin - real_y_pred)))

        return numpy.mean(loss)


def contrastive_loss(
    y_true: numpy.ndarray | list = None,
    y_pred: numpy.ndarray | list = None,
    margin: float = 0.2,
) -> ContrastiveLoss | numpy.ndarray:
    """Contrastive Loss is commonly used in metric learning tasks to learn an embedding space where similar pairs of examples are closer, and dissimilar pairs are farther apart. It's typically used for Siamese Networks or triplet-based architectures.

    Args:
        y_true (numpy.ndarray|list): True Values
        y_pred (numpy.ndarray|list): Predicted Values from network
        margin (float):Margin for dissimilar pairs (default 1.0)

    Raises:
        ValueError: when y_true is not type (list , numpy.ndarray)
        ValueError: when y_pred is not type (list , numpy.ndarray)
        ValueError: when margin is not type (float)

    Returns:
        numpy.ndarray: the Contrastive loss value (scalar).
    """
    if y_true is None and y_pred is None:
        return ContrastiveLoss(margin=margin)

    return ContrastiveLoss(margin=margin).calc(
        y_true=y_true,
        y_pred=y_pred,
    )
