import numpy

from .activation_function import ActivationFunction

from ..errors import ActivationErrors

class HardSwish(ActivationFunction):
    """The Hard Swish function is a piecewise approximation of the Swish function.
    The clip function ensures that the value of x+366x+3​ is bounded between 0 and 1, which makes the function computationally efficient, particularly for hardware accelerators like mobile devices.

    Args:
        x (numpy.ndarray | list): input to the function

    Raises:
        ActivationErrors: ActivationErrors: if `x` is not in (list, numpy.ndarray) types

    Returns:
        numpy.ndarray: output of the `HardSwish(x)`
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: numpy.ndarray | list) -> numpy.ndarray:
        """The Hard Swish function is a piecewise approximation of the Swish function.
        The clip function ensures that the value of x+366x+3​ is bounded between 0 and 1, which makes the function computationally efficient, particularly for hardware accelerators like mobile devices.

        Args:
            x (numpy.ndarray | list): input to the function

        Raises:
            ActivationErrors: ActivationErrors: if `x` is not in (list, numpy.ndarray) types

        Returns:
            numpy.ndarray: output of the `HardSwish(x)`
        """
        if not isinstance(x, list) and not isinstance(x, numpy.ndarray):
            raise ActivationErrors(f"{type(self)}.forward 'x' argument must be type of (list , numpy.ndarray) , {type(x)} passed !")

        real_x = x if isinstance(x, numpy.ndarray) else numpy.array(x)

        return real_x * numpy.clip((real_x + 3) / 6, 0, 1)

    def derivate(self, x: numpy.ndarray | list) -> numpy.ndarray:
        """The derivative of the Hard Swish function is a piecewise linear function

        The derivative is linear in the range [−3,3][−3,3] and 0 outside this range, since the function is flat (constant) for values outside of this interval due to the clipping operation.

        Args:
            x (numpy.ndarray | list): input ot the function

        Raises:
            ActivationErrors: ActivationErrors: if `x` is not in (list, numpy.ndarray) types

        Returns:
            numpy.ndarray: derivation of `HardSwish(x)`
        """
        if not isinstance(x, list) and not isinstance(x, numpy.ndarray):
            raise ActivationErrors(f"{type(self)}.forward 'x' argument must be type of (list , numpy.ndarray) , {type(x)} passed !")

        real_x = x if isinstance(x, numpy.ndarray) else numpy.array(x)
        return numpy.where((real_x >= -3) & (real_x <= 3), (real_x + 3) / 6 + 0.5, 0)


def hardswish(x: numpy.ndarray | list = None) -> HardSwish | numpy.ndarray:
    """The Hard Swish function is a piecewise approximation of the Swish function.
    The clip function ensures that the value of x+366x+3​ is bounded between 0 and 1, which makes the function computationally efficient, particularly for hardware accelerators like mobile devices.

    Args:
        x (numpy.ndarray | list): input to the function

    Raises:
        ActivationErrors: ActivationErrors: if `x` is not in (list, numpy.ndarray) types

    Returns:
        numpy.ndarray: output of the `HardSwish(x)`
        HardSwish: if `x == None` the object of `HardSwish` will return
    """
    return HardSwish() if x is None else HardSwish().forward(x)


def hardswish_derivation(x: numpy.ndarray | list) -> numpy.ndarray:
    """The derivative of the Hard Swish function is a piecewise linear function

    The derivative is linear in the range [−3,3][−3,3] and 0 outside this range, since the function is flat (constant) for values outside of this interval due to the clipping operation.

    Args:
        x (numpy.ndarray | list): input ot the function

    Raises:
        ActivationErrors: ActivationErrors: if `x` is not in (list, numpy.ndarray) types

    Returns:
        numpy.ndarray: derivation of `HardSwish(x)`
    """
    return HardSwish().derivate(x)
