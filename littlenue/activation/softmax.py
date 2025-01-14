import numpy

from .activation_function import ActivationFunction


class Softmax(ActivationFunction):
    """Softmax transforms the raw input values (often referred to as logits) into probabilities. The output values are in the range `[0,1]` and sum up to 1, which is useful for multiclass classification problems.

    The implementation uses a numerical stability trick by subtracting the maximum value from the input vector x before applying the exponential function. This ensures that the exponentials do not overflow for large values.

    Args:
        x (numpy.ndarray | list): input to the function

    Raises:
        ValueError: if `x` is not in (list, numpy.ndarray) types

    Returns:
        numpy.ndarray: output of the `Softmax(x)`
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: numpy.ndarray | list) -> numpy.ndarray:
        """Softmax transforms the raw input values (often referred to as logits) into probabilities. The output values are in the range `[0,1]` and sum up to 1, which is useful for multiclass classification problems.

        The implementation uses a numerical stability trick by subtracting the maximum value from the input vector x before applying the exponential function. This ensures that the exponentials do not overflow for large values.

        Args:
            x (numpy.ndarray | list): input to the function

        Raises:
            ValueError: if `x` is not in (list, numpy.ndarray) types

        Returns:
            numpy.ndarray: output of the `Softmax(x)`
        """
        if not isinstance(x, list) and not isinstance(x, numpy.ndarray):
            raise ValueError(f"{type(self)}.forward 'x' argument must be type of (list , numpy.ndarray) , {type(x)} passed !")

        real_x = x if isinstance(x, numpy.ndarray) else numpy.array(x)
        exp_x = numpy.exp(real_x - numpy.max(real_x))
        return exp_x / numpy.sum(exp_x)

    def derivate(self, x: numpy.ndarray | list) -> numpy.ndarray:
        """The Jacobian matrix contains the partial derivatives of each softmax output with respect to each input value.
        For each output Softmax(xi), the diagonal elements represent the derivative of that element with respect to the input (i.e., the self-derivative), while the off-diagonal elements represent the cross-derivatives.

        Args:
            x (numpy.ndarray | list): input to the function

        Raises:
            ValueError: if `x` is not in (list, numpy.ndarray) types

        Returns:
            numpy.ndarray: derivation of `Softmax(x)`
        """
        if not isinstance(x, list) and not isinstance(x, numpy.ndarray):
            raise ValueError(f"{type(self)}.forward 'x' argument must be type of (list , numpy.ndarray) , {type(x)} passed !")

        real_x = x if isinstance(x, numpy.ndarray) else numpy.array(x)
        s = self.forward(real_x)  # Apply softmax to get probabilities

        return numpy.diag(s) - numpy.outer(s, s)


def softmax(x: numpy.ndarray | list = None) -> Softmax | numpy.ndarray:
    """Softmax transforms the raw input values (often referred to as logits) into probabilities. The output values are in the range `[0,1]` and sum up to 1, which is useful for multiclass classification problems.

    The implementation uses a numerical stability trick by subtracting the maximum value from the input vector x before applying the exponential function. This ensures that the exponentials do not overflow for large values.

    Args:
        x (numpy.ndarray | list): input to the function

    Raises:
        ValueError: if `x` is not in (list, numpy.ndarray) types

    Returns:
        numpy.ndarray: output of the `Softmax(x)`
        Softmax: if `x== None` object of `Softmax` will return
    """
    return Softmax() if x == None else Softmax().forward(x)


def softmax_derivation(x: numpy.ndarray | list) -> numpy.ndarray:
    """The Jacobian matrix contains the partial derivatives of each softmax output with respect to each input value.
    For each output Softmax(xi), the diagonal elements represent the derivative of that element with respect to the input (i.e., the self-derivative), while the off-diagonal elements represent the cross-derivatives.

    Args:
        x (numpy.ndarray | list): input to the function

    Raises:
        ValueError: if `x` is not in (list, numpy.ndarray) types

    Returns:
        numpy.ndarray: derivation of `Softmax(x)`
    """
    return Softmax().derivate(x)
