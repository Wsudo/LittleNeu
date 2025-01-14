import numpy

from .loss_function import ClassificationLossFunction, MultiLabelLossFunction


class CrossEntropyLoss(ClassificationLossFunction):
    """We clip the predictions to avoid taking the logarithm of 0, which is undefined. The clipping ensures predictions are in the range `[1e−15,1−1e−15]`.
    The binary cross-entropy is calculated for each sample, and the mean is returned.

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
        """We clip the predictions to avoid taking the logarithm of 0, which is undefined. The clipping ensures predictions are in the range `[1e−15,1−1e−15]`.
        The binary cross-entropy is calculated for each sample, and the mean is returned.

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

        real_y_pred = numpy.clip(real_y_pred, 1e-15, 1 - 1e-15)

        # Calculate Binary Cross-Entropy loss
        loss = -(real_y_true * numpy.log(real_y_pred) + (1 - real_y_true) * numpy.log(1 - real_y_pred))

        # Return the mean loss
        return numpy.mean(loss)


class CategoricalCrossEntropyLoss(ClassificationLossFunction):
    """We clip the predicted values to avoid taking the logarithm of 0.
    For each sample, we compute the cross-entropy loss based on the true label and predicted probabilities.
    The mean loss is returned.

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
        """We clip the predicted values to avoid taking the logarithm of 0.
        For each sample, we compute the cross-entropy loss based on the true label and predicted probabilities.
        The mean loss is returned.

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

        # Clip predictions to avoid log(0) which leads to NaN or infinity
        real_y_pred = numpy.clip(real_y_pred, 1e-15, 1 - 1e-15)

        # Calculate Categorical Cross-Entropy loss
        loss = -numpy.sum(real_y_true * numpy.log(real_y_pred), axis=1)

        # Return the mean loss
        return numpy.mean(loss)


class SparseCategoricalCrossEntropyLoss(ClassificationLossFunction):
    """The Sparse Categorical Cross-Entropy Loss is used when the target labels are given as integers (instead of one-hot encoded vectors) in multi-class classification problems. The formula for Sparse Categorical Cross-Entropy is the same as for Categorical Cross-Entropy

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
        """The Sparse Categorical Cross-Entropy Loss is used when the target labels are given as integers (instead of one-hot encoded vectors) in multi-class classification problems. The formula for Sparse Categorical Cross-Entropy is the same as for Categorical Cross-Entropy

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

        real_y_pred = numpy.clip(real_y_pred, 1e-15, 1 - 1e-15)

        # Calculate Sparse Categorical Cross-Entropy loss
        # For each true label y_true, we pick the predicted probability corresponding to that class index
        loss = -numpy.log(real_y_pred[numpy.arange(len(real_y_true)), real_y_true])

        # Return the mean loss
        return numpy.mean(loss)


class MultiLabelCrossEntropyLoss(ClassificationLossFunction, MultiLabelLossFunction):
    """In multi-label classification, each input can be assigned multiple labels (as opposed to single-class assignments in multi-class classification). Binary Cross-Entropy Loss (for multi-label classification) is computed for each class independently, treating the problem as a set of binary classification tasks.

    For multi-label classification, each label is independent, and the loss function is calculated as the binary cross-entropy for each label, then averaged over all labels.

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
        """In multi-label classification, each input can be assigned multiple labels (as opposed to single-class assignments in multi-class classification). Binary Cross-Entropy Loss (for multi-label classification) is computed for each class independently, treating the problem as a set of binary classification tasks.

        For multi-label classification, each label is independent, and the loss function is calculated as the binary cross-entropy for each label, then averaged over all labels.

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

        # Clip predictions to avoid log(0) which leads to NaN or infinity
        real_y_pred = numpy.clip(real_y_pred, 1e-15, 1 - 1e-15)

        # Calculate Binary Cross-Entropy loss for multi-label classification
        loss = -numpy.mean(numpy.sum(real_y_true * numpy.log(real_y_pred) + (1 - real_y_true) * numpy.log(1 - real_y_pred), axis=1))

        return loss


class BinaryCrossEntropyLoss(CrossEntropyLoss):
    """We clip the predictions to avoid taking the logarithm of 0, which is undefined. The clipping ensures predictions are in the range `[1e−15,1−1e−15]`.
    The binary cross-entropy is calculated for each sample, and the mean is returned.

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


class SCCrossEntropyLoss(SparseCategoricalCrossEntropyLoss):
    """The Sparse Categorical Cross-Entropy Loss is used when the target labels are given as integers (instead of one-hot encoded vectors) in multi-class classification problems. The formula for Sparse Categorical Cross-Entropy is the same as for Categorical Cross-Entropy

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


def cross_entropy_loss(
    y_true: numpy.ndarray | list = None,
    y_pred: numpy.ndarray | list = None,
) -> CrossEntropyLoss | numpy.ndarray:
    """We clip the predictions to avoid taking the logarithm of 0, which is undefined. The clipping ensures predictions are in the range `[1e−15,1−1e−15]`.
    The binary cross-entropy is calculated for each sample, and the mean is returned.

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
    return CrossEntropyLoss() if y_true is None and y_pred is None else CrossEntropyLoss().calc(y_true, y_pred)


def binary_cross_entropy_loss(
    y_true: numpy.ndarray | list = None,
    y_pred: numpy.ndarray | list = None,
) -> BinaryCrossEntropyLoss | numpy.ndarray:
    """We clip the predictions to avoid taking the logarithm of 0, which is undefined. The clipping ensures predictions are in the range `[1e−15,1−1e−15]`.
    The binary cross-entropy is calculated for each sample, and the mean is returned.

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
    return BinaryCrossEntropyLoss() if y_true is None and y_pred is None else BinaryCrossEntropyLoss().calc(y_true, y_pred)


def categorical_cross_entropy_loss(
    y_true: numpy.ndarray | list = None,
    y_pred: numpy.ndarray | list = None,
) -> CategoricalCrossEntropyLoss | numpy.ndarray:
    """We clip the predicted values to avoid taking the logarithm of 0.
    For each sample, we compute the cross-entropy loss based on the true label and predicted probabilities.
    The mean loss is returned.

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
    return CategoricalCrossEntropyLoss() if y_true is None and y_pred is None else CategoricalCrossEntropyLoss().calc(y_true, y_pred)


def sparse_categorical_cross_entropy_loss(
    y_true: numpy.ndarray | list = None,
    y_pred: numpy.ndarray | list = None,
) -> SparseCategoricalCrossEntropyLoss | numpy.ndarray:
    """The Sparse Categorical Cross-Entropy Loss is used when the target labels are given as integers (instead of one-hot encoded vectors) in multi-class classification problems. The formula for Sparse Categorical Cross-Entropy is the same as for Categorical Cross-Entropy

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
    return SparseCategoricalCrossEntropyLoss() if y_true is None and y_pred is None else SparseCategoricalCrossEntropyLoss().calc(y_true, y_pred)


def sc_cross_entropy_loss(
    y_true: numpy.ndarray | list = None,
    y_pred: numpy.ndarray | list = None,
) -> SCCrossEntropyLoss | numpy.ndarray:
    """The Sparse Categorical Cross-Entropy Loss is used when the target labels are given as integers (instead of one-hot encoded vectors) in multi-class classification problems. The formula for Sparse Categorical Cross-Entropy is the same as for Categorical Cross-Entropy

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
    return SCCrossEntropyLoss() if y_true is None and y_pred is None else SCCrossEntropyLoss().calc(y_true, y_pred)


def multilabel_cross_entropy_loss(
    y_true: numpy.ndarray | list = None,
    y_pred: numpy.ndarray | list = None,
) -> MultiLabelCrossEntropyLoss | numpy.ndarray:
    """In multi-label classification, each input can be assigned multiple labels (as opposed to single-class assignments in multi-class classification). Binary Cross-Entropy Loss (for multi-label classification) is computed for each class independently, treating the problem as a set of binary classification tasks.

    For multi-label classification, each label is independent, and the loss function is calculated as the binary cross-entropy for each label, then averaged over all labels.

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
    return MultiLabelCrossEntropyLoss() if y_true is None and y_pred is None else MultiLabelCrossEntropyLoss().calc(y_true, y_pred)
