import numpy

from .activation_function import ActivationFunction

class Sigmoid(ActivationFunction):

    def __init__(self):
        super().__init__()

    def forward(self, x: numpy.ndarray|list) -> numpy.ndarray:

        if not isinstance(x , list) and not isinstance(x , numpy.ndarray):
            raise ValueError(f"{type(self)}.forward 'x' argument must be type of (list , numpy.ndarray) , {type(x)} passed !")

        return 1 / (1 + numpy.exp(-(x if isinstance(x , numpy.ndarray) else numpy.array(x))))
    
    def derivate(self, x: numpy.ndarray|list) -> numpy.ndarray:

        if not isinstance(x , list) and not isinstance(x , numpy.ndarray):
            raise ValueError(f"{type(self)}.derivate 'x' argument must be type of (list , numpy.ndarray) , {type(x)} passed !")
        
        sig_x = self.forward(x)

        return sig_x * (1 - sig_x)
    

def sigmoid(x: numpy.ndarray|list = None) -> Sigmoid|numpy.ndarray:
    return Sigmoid() if x == None else Sigmoid().forward(x)


def sigmoid_derivate(x: numpy.ndarray|list) ->numpy.ndarray:
    return Sigmoid().derivate(x)

