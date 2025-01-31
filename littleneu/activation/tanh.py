import numpy

from .activation_function import ActivationFunction

from ..errors import ActivationErrors

class Tanh(ActivationFunction):
    """The tanh function outputs values in the range `(−1,1)`, making it centered around zero, which can be beneficial in some network architectures.

    Args:
        x (numpy.ndarray | list): input to the function

    Raises:
        ActivationErrors: if `x` is not in (list, numpy.ndarray) types

    Returns:
        numpy.ndarray: output of tanh(`x`)
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: numpy.ndarray | list) -> numpy.ndarray:
        """The tanh function outputs values in the range `(−1,1)`, making it centered around zero, which can be beneficial in some network architectures.

        Args:
            x (numpy.ndarray | list): input to the function

        Raises:
            ActivationErrors: if `x` is not in (list, numpy.ndarray) types

        Returns:
            numpy.ndarray: output of tanh(`x`)
        """
        if not isinstance(x, list) and not isinstance(x, numpy.ndarray):
            raise ActivationErrors(f"{type(self)}.forward 'x' argument must be type of (list , numpy.ndarray) , {type(x)} passed !")

        return numpy.tanh(x if isinstance(x, numpy.ndarray) else numpy.array(x))

    def derivate(self, x: numpy.ndarray | list) -> numpy.ndarray:
        """The derivative is used during the backpropagation process in neural networks to update the weights.

        Args:
            x (numpy.ndarray | list): the input to the function

        Raises:
            ActivationErrors: if `x` is not in (list, numpy.ndarray) types

        Returns:
            numpy.ndarray: derivation of the `tanh(x)` = `1 - tanh(x)**2`
        """
        return 1 - numpy.tanh(x) ** 2


def tanh(x: numpy.ndarray | list) -> Tanh | numpy.ndarray:
    """The tanh function outputs values in the range `(−1,1)`, making it centered around zero, which can be beneficial in some network architectures.

    Args:
        x (numpy.ndarray | list): input to the function

    Raises:
        ActivationErrors: if `x` is not in (list, numpy.ndarray) types
        Tanh: if `x == None` the object of `Tanh` will return

    Returns:
        numpy.ndarray: output of tanh(`x`)
    """
    return Tanh() if x is None else Tanh().forward(x)


def tanh_derivation(x: numpy.ndarray | list) -> numpy.ndarray:
    """The derivative is used during the backpropagation process in neural networks to update the weights.

    Args:
        x (numpy.ndarray | list): the input to the function

    Raises:
        ActivationErrors: if `x` is not in (list, numpy.ndarray) types

    Returns:
        numpy.ndarray: derivation of the `tanh(x)` = `1 - tanh(x)**2`
    """
    return Tanh().derivate(x)
