import numpy

from .activation_function import ActivationFunction


class PRelu(ActivationFunction):
    """In PReLU, the parameter αα (which is typically learned during training) is a constant that defines the slope for negative inputs. This allows the model to adaptively learn the best slope for negative values.

    Args:
        x (numpy.ndarray | list): the input to the function
        alpha (float): the `α(alpha)` value , `(commonly 0.1)`.

    Raises:
        ValueError: if `x` is not in (list, numpy.ndarray) types

    Returns:
        numpy.ndarray: output of the function `PRelu(x)`
    """

    def __init__(self, alpha: float = 0.1):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: numpy.ndarray | list) -> numpy.ndarray:
        """In PReLU, the parameter αα (which is typically learned during training) is a constant that defines the slope for negative inputs. This allows the model to adaptively learn the best slope for negative values.

        Args:
            x (numpy.ndarray | list): the input to the function
            alpha (float): the `α(alpha)` value , `(commonly 0.1)`.

        Raises:
            ValueError: if `x` is not in (list, numpy.ndarray) types

        Returns:
            numpy.ndarray: output of the function `PRelu(x)`
        """
        if not isinstance(x, list) and not isinstance(x, numpy.ndarray):
            raise ValueError(f"{type(self)}.forward 'x' argument must be type of (list , numpy.ndarray) , {type(x)} passed !")

        real_x = x if isinstance(x, numpy.ndarray) else numpy.array(x)
        return numpy.where(real_x > 0, real_x, self.alpha * real_x)

    def derivate(self, x: numpy.ndarray | list) -> numpy.ndarray:
        """The derivative is used in backpropagation to compute gradients. For positive values of xx, the derivative is 1, and for negative values, it is α.

        Args:
            x (numpy.ndarray | list): the input to the function

        Raises:
            ValueError: if `x` is not in (list, numpy.ndarray) types

        Returns:
            numpy.ndarray: `derivation` of `PRelu(x)`
        """
        if not isinstance(x, list) and not isinstance(x, numpy.ndarray):
            raise ValueError(f"{type(self)}.forward 'x' argument must be type of (list , numpy.ndarray) , {type(x)} passed !")

        real_x = x if isinstance(x, numpy.ndarray) else numpy.array(x)
        return numpy.where(real_x > 0, 1, self.alpha)


class ParametricRelu(PRelu):
    """In PReLU, the parameter αα (which is typically learned during training) is a constant that defines the slope for negative inputs. This allows the model to adaptively learn the best slope for negative values.

    Args:
        x (numpy.ndarray | list): the input to the function
        alpha (float): the `α(alpha)` value , `(commonly 0.1)`.

    Raises:
        ValueError: if `x` is not in (list, numpy.ndarray) types

    Returns:
        numpy.ndarray: output of the function `PRelu(x)`
    """

    pass


def prelu(x: numpy.ndarray | list = None, alpha: float = 0.1) -> PRelu | numpy.ndarray:
    """In PReLU, the parameter αα (which is typically learned during training) is a constant that defines the slope for negative inputs. This allows the model to adaptively learn the best slope for negative values.

    Args:
        x (numpy.ndarray | list): the input to the function
        alpha (float): the `α(alpha)` value , `(commonly 0.1)`.

    Raises:
        ValueError: if `x` is not in (list, numpy.ndarray) types

    Returns:
        numpy.ndarray: output of the function `PRelu(x)`
    """
    return PRelu(alpha=alpha) if x is None else PRelu(alpha=alpha).forward(x)


def prelu_derivation(x: numpy.ndarray | list = None, alpha: float = 0.1) -> numpy.ndarray:
    """The derivative is used in backpropagation to compute gradients. For positive values of xx, the derivative is 1, and for negative values, it is α.

    Args:
        x (numpy.ndarray | list): the input to the function
        alpha (float): the `α(alpha)` value , `(commonly 0.1)`.

    Raises:
        ValueError: if `x` is not in (list, numpy.ndarray) types

    Returns:
        numpy.ndarray: `derivation` of `PRelu(x)`
    """
    return PRelu(alpha=alpha).derivate(x)


def parametric_relu(x: numpy.ndarray | list = None, alpha: float = 0.1) -> PRelu | numpy.ndarray:
    """In PReLU, the parameter αα (which is typically learned during training) is a constant that defines the slope for negative inputs. This allows the model to adaptively learn the best slope for negative values.

    Args:
        x (numpy.ndarray | list): the input to the function
        alpha (float): the `α(alpha)` value , `(commonly 0.1)`.

    Raises:
        ValueError: if `x` is not in (list, numpy.ndarray) types

    Returns:
        numpy.ndarray: output of the function `PRelu(x)`
    """
    return PRelu(alpha=alpha) if x is None else PRelu(alpha=alpha).forward(x)


def parametric_relu_derivation(x: numpy.ndarray | list = None, alpha: float = 0.1) -> numpy.ndarray:
    """The derivative is used in backpropagation to compute gradients. For positive values of xx, the derivative is 1, and for negative values, it is α.

    Args:
        x (numpy.ndarray | list): the input to the function
        alpha (float): the `α(alpha)` value , `(commonly 0.1)`.

    Raises:
        ValueError: if `x` is not in (list, numpy.ndarray) types

    Returns:
        numpy.ndarray: `derivation` of `PRelu(x)`
    """
    return PRelu(alpha=alpha).derivate(x)
