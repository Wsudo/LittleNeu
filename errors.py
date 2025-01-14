
class LittleNeuErrors(Exception , BaseException):
    pass

class NeuralNetworksErrors(LittleNeuErrors):
    pass

class ActivationFunctionErrors(NeuralNetworksErrors):
    pass

class InternalErrors(LittleNeuErrors):
    pass

class LittleNeuNotImplementedError(LittleNeuErrors):
    pass