import numpy

from errors import LittleNeuNotImplementedError

class ActivationFunction:

    def forward(self , x:numpy.ndarray|list)->numpy.ndarray:
        return LittleNeuNotImplementedError(f"forward method not implemented for {type(self)} activation function.")
    
    def derivate(self , x: numpy.ndarray|list) -> numpy.ndarray:
        return LittleNeuNotImplementedError(f"derivate method not implemented for {type(self)} activation function.")
