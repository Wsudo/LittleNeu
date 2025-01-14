import numpy

from .activation_function import ActivationFunction

from .sigmoid import Sigmoid


class Swish(ActivationFunction):
    """The Swish function is a smooth, non-monotonic function and has been shown to improve the performance of deep networks in some cases.

    Args:
        x (numpy.ndarray | list): input to the function

    Raises:
        ValueError: if `x` is not in (list, numpy.ndarray) types

    Returns:
        numpy.ndarray: output of the `Swish(x)`
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: numpy.ndarray | list) -> numpy.ndarray:
        """The Swish function is a smooth, non-monotonic function and has been shown to improve the performance of deep networks in some cases.

        Args:
            x (numpy.ndarray | list): input to the function

        Raises:
            ValueError: if `x` is not in (list, numpy.ndarray) types

        Returns:
            numpy.ndarray: output of the `Swish(x)`
        """
        if not isinstance(x, list) and not isinstance(x, numpy.ndarray):
            raise ValueError(f"{type(self)}.forward 'x' argument must be type of (list , numpy.ndarray) , {type(x)} passed !")

        real_x = x if isinstance(x, numpy.ndarray) else numpy.array(x)

        return real_x * Sigmoid().forward(real_x)

    def derivate(self, x: numpy.ndarray | list) -> numpy.ndarray:
        """where the derivative of σ(x) is σ(x)⋅(1−σ(x)).

        Args:
            x (numpy.ndarray | list): input to the function

        Raises:
            ValueError: if `x` is not in (list, numpy.ndarray) types

        Returns:
            numpy.ndarray: derivation of the `Swish(x)`
        """
        if not isinstance(x, list) and not isinstance(x, numpy.ndarray):
            raise ValueError(f"{type(self)}.forward 'x' argument must be type of (list , numpy.ndarray) , {type(x)} passed !")

        real_x = x if isinstance(x, numpy.ndarray) else numpy.array(x)
        sig = Sigmoid.forward(real_x)
        return sig + real_x * sig * (1 - sig)


def swish(x: numpy.ndarray | list = None) -> Swish | numpy.ndarray:
    """The Swish function is a smooth, non-monotonic function and has been shown to improve the performance of deep networks in some cases.

    Args:
        x (numpy.ndarray | list): input to the function

    Raises:
        ValueError: if `x` is not in (list, numpy.ndarray) types

    Returns:
        numpy.ndarray: output of the `Swish(x)`
        Swish: if `x==None` the object of `Swish` will return
    """
    return Swish() if x == None else Swish().forward()


def swish_derivation(x: numpy.ndarray | list = None) -> numpy.ndarray:
    """where the derivative of σ(x) is σ(x)⋅(1−σ(x)).

    Args:
        x (numpy.ndarray | list): input to the function

    Raises:
        ValueError: if `x` is not in (list, numpy.ndarray) types

    Returns:
        numpy.ndarray: derivation of the `Swish(x)`
    """
    return Swish().derivate(x)
