import numpy

from .activation_function import ActivationFunction


class Relu(ActivationFunction):
    """The ReLU activation function sets all negative values to zero and keeps all positive values as they are. It is one of the most widely used activation functions in neural networks due to its simplicity and effectiveness.

    Args:
        x (numpy.ndarray | list): input to the function

    Raises:
        ValueError: if `x` is not in (list, numpy.ndarray) types

    Returns:
        numpy.ndarray: output of the `Relu(x)` == `max(0,x)`
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: numpy.ndarray | list) -> numpy.ndarray:
        """The ReLU activation function sets all negative values to zero and keeps all positive values as they are. It is one of the most widely used activation functions in neural networks due to its simplicity and effectiveness.

        Args:
            x (numpy.ndarray | list): input to the function

        Raises:
            ValueError: if `x` is not in (list, numpy.ndarray) types

        Returns:
            numpy.ndarray: output of the `Relu(x)` == `max(0,x)`
        """
        if not isinstance(x, list) and not isinstance(x, numpy.ndarray):
            raise ValueError(f"{type(self)}.derivate 'x' argument must be type of (list , numpy.ndarray) , {type(x)} passed !")

        return numpy.maximum(0, x if isinstance(x, numpy.ndarray) else numpy.array(x))

    def derivate(self, x: numpy.ndarray | list) -> numpy.ndarray:
        """The derivative is used during backpropagation for updating the model's weights. For positive inputs, the derivative is 1 (indicating that the output changes at the same rate as the input), and for non-positive inputs, the derivative is 0 (indicating that the output doesn't change).

        Args:
            x (numpy.ndarray | list): input to the function

        Raises:
            ValueError: if `x` is not in (list, numpy.ndarray) types

        Returns:
            numpy.ndarray: derivation of the `Relu(x)`
        """
        if not isinstance(x, list) and not isinstance(x, numpy.ndarray):
            raise ValueError(f"{type(self)}.derivate 'x' argument must be type of (list , numpy.ndarray) , {type(x)} passed !")

        return numpy.where((x if isinstance(x, numpy.ndarray) else numpy.array(x)) > 0, 1, 0)


def relu(x: numpy.ndarray | list = None) -> Relu | numpy.ndarray:
    """The ReLU activation function sets all negative values to zero and keeps all positive values as they are. It is one of the most widely used activation functions in neural networks due to its simplicity and effectiveness.

    Args:
        x (numpy.ndarray | list): input to the function

    Raises:
        ValueError: if `x` is not in (list, numpy.ndarray) types

    Returns:
        numpy.ndarray: output of the `Relu(x)` == `max(0,x)`
    """
    return Relu() if x == None else Relu().forward(x)


def relu_derivation(x: numpy.ndarray | list) -> numpy.ndarray:
    """The derivative is used during backpropagation for updating the model's weights. For positive inputs, the derivative is 1 (indicating that the output changes at the same rate as the input), and for non-positive inputs, the derivative is 0 (indicating that the output doesn't change).

    Args:
        x (numpy.ndarray | list): input to the function

    Raises:
        ValueError: if `x` is not in (list, numpy.ndarray) types

    Returns:
        numpy.ndarray: derivation of the `Relu(x)`
    """
    return Relu().derivate(x)
