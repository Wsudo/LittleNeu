import numpy

from .loss_function import MetricLearningLossFunction


class TripletLoss(MetricLearningLossFunction):
    """Triplet Loss is commonly used in metric learning to learn embeddings that minimize the distance between similar samples (anchor and positive) while maximizing the distance between dissimilar samples (anchor and negative).

    Args:
        anchor (numpy.ndarray|list): Embedding of the anchor sample (numpy array)
        positive (numpy.ndarray|list): Embedding of the positive sample (numpy array)
        negative (numpy.ndarray|list): Embedding of the negative sample (numpy array)
        alpha (float):Margin between positive and negative distances (default 0.2)

    Raises:
        ValueError: when anchor is not type (list , numpy.ndarray)
        ValueError: when positive is not type (list , numpy.ndarray)
        ValueError: when negative is not type (list , numpy.ndarray)
        ValueError: when alpha is not type (float)

    Returns:
        numpy.ndarray: the triplet loss value (scalar).
    """

    def __init__(
        self,
        alpha: float = 0.2,
    ):
        super().__init__()

        self.alpha = alpha

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
        anchor: numpy.ndarray | list,
        positive: numpy.ndarray | list,
        negative: numpy.ndarray | list,
    ) -> numpy.ndarray:
        """Triplet Loss is commonly used in metric learning to learn embeddings that minimize the distance between similar samples (anchor and positive) while maximizing the distance between dissimilar samples (anchor and negative).

        Args:
            anchor (numpy.ndarray|list): Embedding of the anchor sample (numpy array)
            positive (numpy.ndarray|list): Embedding of the positive sample (numpy array)
            negative (numpy.ndarray|list): Embedding of the negative sample (numpy array)
            alpha (float):Margin between positive and negative distances (default 0.2)

        Raises:
            ValueError: when anchor is not type (list , numpy.ndarray)
            ValueError: when positive is not type (list , numpy.ndarray)
            ValueError: when negative is not type (list , numpy.ndarray)
            ValueError: when alpha is not type (float)

        Returns:
            numpy.ndarray: the triplet loss value (scalar).
        """
        if not isinstance(anchor, numpy.ndarray) and not isinstance(anchor, list):
            raise ValueError(f"{type(self)}.calc 'anchor' argument must be type (list , numpy.ndarray) , {type(anchor)} passed !")
        if not isinstance(positive, numpy.ndarray) and not isinstance(positive, list):
            raise ValueError(f"{type(self)}.calc 'positive' argument must be type (list , numpy.ndarray) , {type(positive)} passed !")
        if not isinstance(negative, numpy.ndarray) and not isinstance(negative, list):
            raise ValueError(f"{type(self)}.calc 'negative' argument must be type (list , numpy.ndarray) , {type(negative)} passed !")

        real_anchor = anchor if isinstance(anchor, numpy.ndarray) else numpy.array(anchor)
        real_positive = positive if isinstance(positive, numpy.ndarray) else numpy.array(positive)
        real_negative = negative if isinstance(negative, numpy.ndarray) else numpy.array(negative)

        # Compute the Euclidean distances
        d_ap = self.euclidean_distance(anchor, positive)  # Distance between anchor and positive
        d_an = self.euclidean_distance(anchor, negative)  # Distance between anchor and negative

        # Compute the triplet loss
        loss = numpy.maximum(d_ap - d_an + self.alpha, 0)

        return loss


def triplet_loss(
    anchor: numpy.ndarray | list = None,
    positive: numpy.ndarray | list = None,
    negative: numpy.ndarray | list = None,
    alpha: float = 0.2,
) -> TripletLoss | numpy.ndarray:
    """Triplet Loss is commonly used in metric learning to learn embeddings that minimize the distance between similar samples (anchor and positive) while maximizing the distance between dissimilar samples (anchor and negative).

    Args:
        anchor (numpy.ndarray|list): Embedding of the anchor sample (numpy array)
        positive (numpy.ndarray|list): Embedding of the positive sample (numpy array)
        negative (numpy.ndarray|list): Embedding of the negative sample (numpy array)
        alpha (float):Margin between positive and negative distances (default 0.2)

    Raises:
        ValueError: when anchor is not type (list , numpy.ndarray)
        ValueError: when positive is not type (list , numpy.ndarray)
        ValueError: when negative is not type (list , numpy.ndarray)
        ValueError: when alpha is not type (float)

    Returns:
        numpy.ndarray: the triplet loss value (scalar).
    """
    if anchor is None and positive is None and negative is None:
        return TripletLoss(alpha=alpha)

    return TripletLoss(alpha=alpha).calc(
        anchor=anchor,
        positive=positive,
        negative=negative,
    )
