import numpy

from .activation_function import ActivationFunction


class HardSigmoid(ActivationFunction):
    """The Hard Sigmoid function is a piecewise linear approximation of the Sigmoid function, which makes it computationally efficient.

    Args:
        x (numpy.ndarray | list): input to the function

    Raises:
        ValueError: ValueError: if `x` is not in (list, numpy.ndarray) types

    Returns:
        numpy.ndarray: output of `HardSigmoid(x)`
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: numpy.ndarray | list) -> numpy.ndarray:
        """The Hard Sigmoid function is a piecewise linear approximation of the Sigmoid function, which makes it computationally efficient.

        Args:
            x (numpy.ndarray | list): input to the function

        Raises:
            ValueError: ValueError: if `x` is not in (list, numpy.ndarray) types

        Returns:
            numpy.ndarray: output of `HardSigmoid(x)`
        """
        if not isinstance(x, list) and not isinstance(x, numpy.ndarray):
            raise ValueError(f"{type(self)}.forward 'x' argument must be type of (list , numpy.ndarray) , {type(x)} passed !")

        real_x = x if isinstance(x, numpy.ndarray) else numpy.array(x)
        return numpy.clip(0.2 * real_x + 0.5, 0, 1)

    def derivate(self, x: numpy.ndarray | list) -> numpy.ndarray:
        """The derivative of the Hard Sigmoid function is constant, and it is 0.2 for inputs within a certain range and 0 elsewhere.

        Args:
            x (numpy.ndarray | list): input to the function

        Raises:
            ValueError: ValueError: if `x` is not in (list, numpy.ndarray) types

        Returns:
            numpy.ndarray: derivation of `HardSigmoid(x)`
        """
        if not isinstance(x, list) and not isinstance(x, numpy.ndarray):
            raise ValueError(f"{type(self)}.forward 'x' argument must be type of (list , numpy.ndarray) , {type(x)} passed !")

        real_x = x if isinstance(x, numpy.ndarray) else numpy.array(x)
        return numpy.where((real_x >= -2.5) & (real_x <= 2.5), 0.2, 0)


def hardsigmoid(x: numpy.ndarray | list = None) -> HardSigmoid | numpy.ndarray:
    """The Hard Sigmoid function is a piecewise linear approximation of the Sigmoid function, which makes it computationally efficient.

    Args:
        x (numpy.ndarray | list): input to the function

    Raises:
        ValueError: ValueError: if `x` is not in (list, numpy.ndarray) types

    Returns:
        numpy.ndarray: output of `HardSigmoid(x)`
        HardSigmoid: if `x == None` the object of `HardSigmoid` will return
    """
    return HardSigmoid() if x == None else HardSigmoid().forward(x)


def hardsigmoid_derivation(x: numpy.ndarray | list) -> numpy.ndarray:
    """The derivative of the Hard Sigmoid function is constant, and it is 0.2 for inputs within a certain range and 0 elsewhere.

    Args:
        x (numpy.ndarray | list): input to the function

    Raises:
        ValueError: ValueError: if `x` is not in (list, numpy.ndarray) types

    Returns:
        numpy.ndarray: derivation of `HardSigmoid(x)`
    """
    return HardSigmoid().derivate(x)
