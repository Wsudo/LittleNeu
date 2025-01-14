import numpy

from .activation_function import ActivationFunction


class LeakyRelu(ActivationFunction):
    """Here, `α(alpha)` is a small positive constant `(commonly 0.01)`. The function allows a small negative slope for inputs less than `0`, preventing `"dead neurons"` (which can happen with the standard ReLU).

    Args:
        x (numpy.ndarray | list): the input to the function
        alpha (float): the `α(alpha)` value , `(commonly 0.01)`.

    Raises:
        ValueError: if `x` is not in (list, numpy.ndarray) types

    Returns:
        numpy.ndarray: output of the function `LeakyRelu(x)`
    """

    def __init__(self, alpha=0.01):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: numpy.ndarray | list) -> numpy.ndarray:
        """Here, `α(alpha)` is a small positive constant `(commonly 0.01)`. The function allows a small negative slope for inputs less than `0`, preventing `"dead neurons"` (which can happen with the standard ReLU).

        Args:
            x (numpy.ndarray | list): the input to the function
            alpha (float): the `α(alpha)` value , `(commonly 0.01)`.

        Raises:
            ValueError: if `x` is not in (list, numpy.ndarray) types

        Returns:
            numpy.ndarray: output of the function `LeakyRelu(x)`
        """
        if not isinstance(x, list) and not isinstance(x, numpy.ndarray):
            raise ValueError(f"{type(self)}.forward 'x' argument must be type of (list , numpy.ndarray) , {type(x)} passed !")

        real_x = x if isinstance(x, numpy.ndarray) else numpy.array(x)
        return numpy.where(real_x > 0, real_x, self.alpha * real_x)

    def derivate(self, x: numpy.ndarray | list) -> numpy.ndarray:
        """The derivative is used during backpropagation to update the weights, where αα controls the slope for negative inputs.

        Args:
            x (numpy.ndarray | list): the input to the function

        Raises:
            ValueError: if `x` is not in (list, numpy.ndarray) types

        Returns:
            numpy.ndarray: `derivation` of `LeakyRelu(x)`
        """
        if not isinstance(x, list) and not isinstance(x, numpy.ndarray):
            raise ValueError(f"{type(self)}.forward 'x' argument must be type of (list , numpy.ndarray) , {type(x)} passed !")

        real_x = x if isinstance(x, numpy.ndarray) else numpy.array(x)
        return numpy.where(real_x > 0, 1, self.alpha)


def leaky_relu(x: numpy.ndarray | list = None, alpha: float = 0.01) -> LeakyRelu | numpy.ndarray:
    """Here, `α(alpha)` is a small positive constant `(commonly 0.01)`. The function allows a small negative slope for inputs less than `0`, preventing `"dead neurons"` (which can happen with the standard ReLU).

    Args:
        x (numpy.ndarray | list): the input to the function
        alpha (float): the `α(alpha)` value , `(commonly 0.01)`.

    Raises:
        ValueError: if `x` is not in (list, numpy.ndarray) types

    Returns:
        numpy.ndarray: output of the function `LeakyRelu(x)`
    """
    return LeakyRelu(alpha=alpha) if x == None else LeakyRelu(alpha=alpha).forward(x)


def leaky_relu_derivation(x: numpy.ndarray | list = None, alpha: float = 0.01) -> numpy.ndarray:
    """The derivative is used during backpropagation to update the weights, where αα controls the slope for negative inputs.

    Args:
        x (numpy.ndarray | list): the input to the function
        alpha (float): the `α(alpha)` value , `(commonly 0.01)`.

    Raises:
        ValueError: if `x` is not in (list, numpy.ndarray) types

    Returns:
        numpy.ndarray: `derivation` of `LeakyRelu(x)`
    """
    return LeakyRelu(alpha=alpha).derivate(x)
