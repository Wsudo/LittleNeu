import numpy

from .loss_function import LossFunction


class CosineSimilarityLoss(LossFunction):
    """Cosine Similarity is the cosine of the angle between two vectors

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

    def cosine_similarity(
        self,
        x: numpy.ndarray | list,
        y: numpy.ndarray | list,
    ) -> numpy.ndarray:
        """Compute the cosine similarity between two vectors x and y.
        Cosine similarity ranges from -1 (completely dissimilar) to 1 (completely similar).

        Args:
            x (numpy.ndarray | list): x value
            y (numpy.ndarray | list): y value

        Returns:
            numpy.ndarray: the cosine_similarity of (x ,y)
        """
        real_x = x if isinstance(x, numpy.ndarray) else numpy.array(x)
        real_y = y if isinstance(y, numpy.ndarray) else numpy.array(y)

        dot_product = numpy.dot(real_x, real_y)  # Dot product of x and y
        norm_x = numpy.linalg.norm(real_x)  # L2 norm (magnitude) of x
        norm_y = numpy.linalg.norm(real_y)  # L2 norm (magnitude) of y

        return dot_product / (norm_x * norm_y)

    def calc(
        self,
        y_true: numpy.ndarray | list,
        y_pred: numpy.ndarray | list,
    ) -> numpy.ndarray:
        """Cosine Similarity is the cosine of the angle between two vectors

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

        dot_product = numpy.dot(real_y_true, real_y_pred)
        norm_true = numpy.linalg.norm(real_y_true)
        norm_pred = numpy.linalg.norm(real_y_pred)

        cosine_similarity = dot_product / (norm_true * norm_pred)

        # Cosine similarity loss is defined as 1 - cosine similarity
        loss = 1 - cosine_similarity

        return loss


def cosine_similarity_loss(
    y_true: numpy.ndarray | list,
    y_pred: numpy.ndarray | list,
) -> CosineSimilarityLoss | numpy.ndarray:
    """Cosine Similarity is the cosine of the angle between two vectors

        y_true (numpy.ndarray | list): true values
        y_pred (numpy.ndarray | list): network generated values

    Raises:
        ValueError: when y_true is not type (list , numpy.ndarray)
        ValueError: when y_pred is not type (list , numpy.ndarray)

    Returns:
        numpy.ndarray: loss between `True Values` and `Predicted Values`
    """
    return CosineSimilarityLoss() if y_true is None and y_pred is None else CosineSimilarityLoss().calc(y_true , y_pred)
