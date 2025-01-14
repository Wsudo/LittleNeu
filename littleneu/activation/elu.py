import numpy

from .activation_function import ActivationFunction


class ELU(ActivationFunction):
    """The ELU activation function is used to allow the model to learn faster and perform better by reducing the bias shift. It has a smooth curve for negative values, unlike ReLU, which cuts off at zero.

    Args:
        x (numpy.ndarray | list): input to the function
        alpha(float) : the alpha value , `commanly(1.0)`

    Raises:
        ValueError: ValueError: if `x` is not in (list, numpy.ndarray) types

    Returns:
        numpy.ndarray: output of the `ELU(x)`
    """

    def __init__(self, alpha: float = 1.0):
        super().__init__()

        self.alpha = alpha

    def forward(self, x: numpy.ndarray | list) -> numpy.ndarray:
        """The ELU activation function is used to allow the model to learn faster and perform better by reducing the bias shift. It has a smooth curve for negative values, unlike ReLU, which cuts off at zero.

        Args:
            x (numpy.ndarray | list): input to the function
            alpha(float) : the alpha value , `commanly(1.0)`

        Raises:
            ValueError: ValueError: if `x` is not in (list, numpy.ndarray) types

        Returns:
            numpy.ndarray: output of the `ELU(x)`
        """
        if not isinstance(x, list) and not isinstance(x, numpy.ndarray):
            raise ValueError(f"{type(self)}.forward 'x' argument must be type of (list , numpy.ndarray) , {type(x)} passed !")

        real_x = x if isinstance(x, numpy.ndarray) else numpy.array(x)
        return numpy.where(real_x > 0, real_x, self.alpha * (numpy.exp(real_x) - 1))

    def derivate(self, x: numpy.ndarray | list) -> numpy.ndarray:
        """The derivative is used in the backpropagation phase of training.

        Args:
            x (numpy.ndarray | list): input to the function
            alpha(float) : the alpha value , `commanly(1.0)`

        Raises:
            ValueError: ValueError: if `x` is not in (list, numpy.ndarray) types

        Returns:
            numpy.ndarray: derivation of `ELU(x)`
        """
        if not isinstance(x, list) and not isinstance(x, numpy.ndarray):
            raise ValueError(f"{type(self)}.forward 'x' argument must be type of (list , numpy.ndarray) , {type(x)} passed !")

        real_x = x if isinstance(x, numpy.ndarray) else numpy.array(x)
        return numpy.where(real_x > 0, 1, self.alpha * numpy.exp(real_x))


class ExponentialLinearUnit(ELU):
    """The ELU activation function is used to allow the model to learn faster and perform better by reducing the bias shift. It has a smooth curve for negative values, unlike ReLU, which cuts off at zero.

    Args:
        x (numpy.ndarray | list): input to the function
        alpha(float) : the alpha value , `commanly(1.0)`

    Raises:
        ValueError: ValueError: if `x` is not in (list, numpy.ndarray) types

    Returns:
        numpy.ndarray: output of the `ELU(x)`
    """

    def __init__(self, alpha: float = 1.0):
        super().__init__(alpha)


def elu(x: numpy.ndarray | list = None, alpha: float = 1.0) -> ELU | numpy.ndarray:
    """The ELU activation function is used to allow the model to learn faster and perform better by reducing the bias shift. It has a smooth curve for negative values, unlike ReLU, which cuts off at zero.

    Args:
        x (numpy.ndarray | list): input to the function
        alpha(float) : the alpha value , `commanly(1.0)`

    Raises:
        ValueError: ValueError: if `x` is not in (list, numpy.ndarray) types

    Returns:
        numpy.ndarray: output of the `ELU(x)`
        ELU: if `x == None` the object of `ELU` will return
    """
    return ELU(alpha=alpha) if x is None else ELU(alpha=alpha).forward(x)


def elu_derivation(x: numpy.ndarray | list, alpha: float = 1.0) -> numpy.ndarray:
    """The derivative is used in the backpropagation phase of training.

    Args:
        x (numpy.ndarray | list): input to the function
        alpha(float) : the alpha value , `commanly(1.0)`

    Raises:
        ValueError: ValueError: if `x` is not in (list, numpy.ndarray) types

    Returns:
        numpy.ndarray: derivation of `ELU(x)`
    """
    return ELU(alpha=alpha).derivate(x)


def exponential_linear_unit(x: numpy.ndarray | list = None, alpha: float = 1.0) -> ELU | numpy.ndarray:
    """The ELU activation function is used to allow the model to learn faster and perform better by reducing the bias shift. It has a smooth curve for negative values, unlike ReLU, which cuts off at zero.

    Args:
        x (numpy.ndarray | list): input to the function
        alpha(float) : the alpha value , `commanly(1.0)`

    Raises:
        ValueError: ValueError: if `x` is not in (list, numpy.ndarray) types

    Returns:
        numpy.ndarray: output of the `ELU(x)`
        ELU: if `x == None` the object of `ELU` will return
    """
    return ExponentialLinearUnit(alpha=alpha) if x is None else ExponentialLinearUnit(alpha=alpha).forward(x)


def exponential_linear_unit_derivation(x: numpy.ndarray | list, alpha: float = 1.0) -> numpy.ndarray:
    """The derivative is used in the backpropagation phase of training.

    Args:
        x (numpy.ndarray | list): input to the function
        alpha(float) : the alpha value , `commanly(1.0)`

    Raises:
        ValueError: ValueError: if `x` is not in (list, numpy.ndarray) types

    Returns:
        numpy.ndarray: derivation of `ELU(x)`
    """
    return ExponentialLinearUnit(alpha=alpha).derivate(x)
