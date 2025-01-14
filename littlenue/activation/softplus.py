import numpy

from .activation_function import ActivationFunction


class Softplus(ActivationFunction):
    """The Softplus function is a smooth approximation of the ReLU function,
    providing a soft, differentiable alternative to ReLU. It outputs values in the range [0,∞), with the function approaching xx for large positive inputs and approaching 0 for large negative inputs.

    Args:
        x (numpy.ndarray | list): input to the function

    Raises:
        ValueError: ValueError: if `x` is not in (list, numpy.ndarray) types

    Returns:
        numpy.ndarray: output of the `SoftPlus(x)`
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: numpy.ndarray | list) -> numpy.ndarray:
        """The Softplus function is a smooth approximation of the ReLU function,
        providing a soft, differentiable alternative to ReLU. It outputs values in the range [0,∞), with the function approaching xx for large positive inputs and approaching 0 for large negative inputs.

        Args:
            x (numpy.ndarray | list): input to the function

        Raises:
            ValueError: ValueError: if `x` is not in (list, numpy.ndarray) types

        Returns:
            numpy.ndarray: output of the `SoftPlus(x)`
        """
        if not isinstance(x, list) and not isinstance(x, numpy.ndarray):
            raise ValueError(f"{type(self)}.forward 'x' argument must be type of (list , numpy.ndarray) , {type(x)} passed !")

        real_x = x if isinstance(x, numpy.ndarray) else numpy.array(x)
        return numpy.log(1 + numpy.exp(real_x))

    def derivate(self, x: numpy.ndarray | list) -> numpy.ndarray:
        """This is equivalent to the Sigmoid function, which ensures that the gradient is bounded between 0 and 1.

        Args:
            x (numpy.ndarray | list): input to the function

        Raises:
            ValueError: ValueError: if `x` is not in (list, numpy.ndarray) types

        Returns:
            numpy.ndarray: derivation of the `SoftPlus(x)`
        """
        if not isinstance(x, list) and not isinstance(x, numpy.ndarray):
            raise ValueError(f"{type(self)}.forward 'x' argument must be type of (list , numpy.ndarray) , {type(x)} passed !")

        real_x = x if isinstance(x, numpy.ndarray) else numpy.array(x)
        return 1 / (1 + numpy.exp(-real_x))


def softplus(x: numpy.ndarray | list = None) -> Softplus | numpy.ndarray:
    """The Softplus function is a smooth approximation of the ReLU function,
    providing a soft, differentiable alternative to ReLU. It outputs values in the range [0,∞), with the function approaching xx for large positive inputs and approaching 0 for large negative inputs.

    Args:
        x (numpy.ndarray | list): input to the function

    Raises:
        ValueError: ValueError: if `x` is not in (list, numpy.ndarray) types

    Returns:
        numpy.ndarray: output of the `SoftPlus(x)`
        Softplus: if `x == None` the object of the `Softplus` will return
    """
    return Softplus() if x == None else Softplus().forward(x)


def softplus_derivation(x: numpy.ndarray | list) -> numpy.ndarray:
    """This is equivalent to the Sigmoid function, which ensures that the gradient is bounded between 0 and 1.

    Args:
        x (numpy.ndarray | list): input to the function

    Raises:
        ValueError: ValueError: if `x` is not in (list, numpy.ndarray) types

    Returns:
        numpy.ndarray: derivation of the `SoftPlus(x)`
    """
    return Softplus().derivate(x)
