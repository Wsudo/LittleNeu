import numpy

from .loss_function import MetricLearningLossFunction

from ..errors import LossErrors

class TripletMarginLoss(MetricLearningLossFunction):
    """Triplet Margin Loss is a loss function commonly used in metric learning tasks, especially in Siamese networks or triplet-based networks. It is designed to learn a feature space where the distance between an anchor and a positive sample (same class) is smaller than the distance between the anchor and a negative sample (different class) by a certain margin.

    Args:
        anchor (numpy.ndarray|list): Embedding of the anchor sample (numpy array)
        positive (numpy.ndarray|list): Embedding of the positive sample (numpy array)
        negative (numpy.ndarray|list): Embedding of the negative sample (numpy array)
        margin (float):Margin for dissimilar pairs (default 1.0)

    Raises:
        LossErrors: when anchor is not type (list , numpy.ndarray)
        LossErrors: when positive is not type (list , numpy.ndarray)
        LossErrors: when negative is not type (list , numpy.ndarray)
        LossErrors: when margin is not type (float)

    Returns:
        numpy.ndarray: the triplet margin loss value (scalar).
    """

    def __init__(
        self,
        margin: float = 1.0,
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
            LossErrors: when x is not type(list,numpy.ndarray)
            LossErrors: when y is not type(list,numpy.ndarray)

        Returns:
            numpy.ndarray: output of `Euclidean distance (x , y)`
        """

        if not isinstance(x, numpy.ndarray) and not isinstance(x, list):
            raise LossErrors(f"{type(self)}.calc 'x' argument must be type (list , numpy.ndarray) , {type(x)} passed !")
        if not isinstance(y, numpy.ndarray) and not isinstance(y, list):
            raise LossErrors(f"{type(self)}.calc 'y' argument must be type (list , numpy.ndarray) , {type(y)} passed !")

        real_x = x if isinstance(x, numpy.ndarray) else numpy.array(x)
        real_y = y if isinstance(y, numpy.ndarray) else numpy.array(y)

        return numpy.linalg.norm(real_x - real_y)

    def calc(
        self,
        anchor: numpy.ndarray | list,
        positive: numpy.ndarray | list,
        negative: numpy.ndarray | list,
    ) -> numpy.ndarray:
        """Triplet Margin Loss is a loss function commonly used in metric learning tasks, especially in Siamese networks or triplet-based networks. It is designed to learn a feature space where the distance between an anchor and a positive sample (same class) is smaller than the distance between the anchor and a negative sample (different class) by a certain margin.

        Args:
            anchor (numpy.ndarray|list): Embedding of the anchor sample (numpy array)
            positive (numpy.ndarray|list): Embedding of the positive sample (numpy array)
            negative (numpy.ndarray|list): Embedding of the negative sample (numpy array)
            margin (float):Margin for dissimilar pairs (default 1.0)

        Raises:
            LossErrors: when anchor is not type (list , numpy.ndarray)
            LossErrors: when positive is not type (list , numpy.ndarray)
            LossErrors: when negative is not type (list , numpy.ndarray)
            LossErrors: when margin is not type (float)

        Returns:
            numpy.ndarray: the triplet margin loss value (scalar).
        """
        if not isinstance(anchor, numpy.ndarray) and not isinstance(anchor, list):
            raise LossErrors(f"{type(self)}.calc 'anchor' argument must be type (list , numpy.ndarray) , {type(anchor)} passed !")
        if not isinstance(positive, numpy.ndarray) and not isinstance(positive, list):
            raise LossErrors(f"{type(self)}.calc 'positive' argument must be type (list , numpy.ndarray) , {type(positive)} passed !")
        if not isinstance(negative, numpy.ndarray) and not isinstance(negative, list):
            raise LossErrors(f"{type(self)}.calc 'negative' argument must be type (list , numpy.ndarray) , {type(negative)} passed !")

        real_anchor = anchor if isinstance(anchor, numpy.ndarray) else numpy.array(anchor)
        real_positive = positive if isinstance(positive, numpy.ndarray) else numpy.array(positive)
        real_negative = negative if isinstance(negative, numpy.ndarray) else numpy.array(negative)

        # Compute the Euclidean distances
        d_ap = self.euclidean_distance(anchor, positive)  # Distance between anchor and positive
        d_an = self.euclidean_distance(anchor, negative)  # Distance between anchor and negative

        # Compute the triplet margin loss
        loss = numpy.maximum(d_ap - d_an + self.margin, 0)

        return loss


def triplet_margin_loss(
    anchor: numpy.ndarray | list = None,
    positive: numpy.ndarray | list = None,
    negative: numpy.ndarray | list = None,
    margin: float = 1.0,
) -> TripletMarginLoss | numpy.ndarray:
    """Triplet Margin Loss is a loss function commonly used in metric learning tasks, especially in Siamese networks or triplet-based networks. It is designed to learn a feature space where the distance between an anchor and a positive sample (same class) is smaller than the distance between the anchor and a negative sample (different class) by a certain margin.

    Args:
        anchor (numpy.ndarray|list): Embedding of the anchor sample (numpy array)
        positive (numpy.ndarray|list): Embedding of the positive sample (numpy array)
        negative (numpy.ndarray|list): Embedding of the negative sample (numpy array)
        margin (float):Margin for dissimilar pairs (default 1.0)

    Raises:
        LossErrors: when anchor is not type (list , numpy.ndarray)
        LossErrors: when positive is not type (list , numpy.ndarray)
        LossErrors: when negative is not type (list , numpy.ndarray)
        LossErrors: when margin is not type (float)

    Returns:
        numpy.ndarray: the triplet margin loss value (scalar).
    """
    if anchor is None and positive is None and negative is None:
        return TripletMarginLoss(margin=margin)

    return TripletMarginLoss(margin=margin).calc(
        anchor=anchor,
        positive=positive,
        negative=negative,
    )
