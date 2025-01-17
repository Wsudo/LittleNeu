import numpy

from .activation_function import ActivationFunction

from ..errors import ActivationErrors

class Sigmoid(ActivationFunction):
    """The Sigmoid function outputs values in the range `(0,1)`, often used for binary classification or as the output layer activation in neural networks.

    Args:
        x (numpy.ndarray | list): a input to the function

    Raises:
        ActivationErrors: if `x` is not in (list, numpy.ndarray) types

    Returns:
        numpy.ndarray: output of sigmoid(`x`) in range `(0,1)`
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: numpy.ndarray | list) -> numpy.ndarray:
        """The Sigmoid function outputs values in the range `(0,1)`, often used for binary classification or as the output layer activation in neural networks.

        Args:
            x (numpy.ndarray | list): a input to the function

        Raises:
            ActivationErrors: if `x` is not in (list, numpy.ndarray) types

        Returns:
            numpy.ndarray: output of sigmoid(`x`) in range `(0,1)`
        """
        if not isinstance(x, list) and not isinstance(x, numpy.ndarray):
            raise ActivationErrors(f"{type(self)}.forward 'x' argument must be type of (list , numpy.ndarray) , {type(x)} passed !")

        return 1 / (1 + numpy.exp(-(x if isinstance(x, numpy.ndarray) else numpy.array(x))))

    def derivate(self, x: numpy.ndarray | list) -> numpy.ndarray:
        """The derivative of the sigmoid function is important during backpropagation to compute the gradient for weight updates.

        Args:
            x (numpy.ndarray | list): a input to the function

        Raises:
            ActivationErrors: if `x` is not in (list, numpy.ndarray) types

        Returns:
            numpy.ndarray: derivation of `x`
        """
        if not isinstance(x, list) and not isinstance(x, numpy.ndarray):
            raise ActivationErrors(f"{type(self)}.derivate 'x' argument must be type of (list , numpy.ndarray) , {type(x)} passed !")

        sig_x = self.forward(x)

        return sig_x * (1 - sig_x)


def sigmoid(x: numpy.ndarray | list = None) -> Sigmoid | numpy.ndarray:
    """The Sigmoid function outputs values in the range `(0,1)`, often used for binary classification or as the output layer activation in neural networks.

    Args:
        x (numpy.ndarray | list): a input to the function

    Raises:
        ActivationErrors: if `x` is not in (list, numpy.ndarray) types

    Returns:
        numpy.ndarray: output of sigmoid(`x`) in range `(0,1)`
        Sigmoid: when no input passed , the object of `Sigmoid` will returned
    """
    return Sigmoid() if x is None else Sigmoid().forward(x)


def sigmoid_derivate(x: numpy.ndarray | list) -> numpy.ndarray:
    """The derivative of the sigmoid function is important during backpropagation to compute the gradient for weight updates.

    Args:
        x (numpy.ndarray | list): a input to the function

    Raises:
        ActivationErrors: if `x` is not in (list, numpy.ndarray) types

    Returns:
        numpy.ndarray: derivation of `x`
    """
    return Sigmoid().derivate(x)
