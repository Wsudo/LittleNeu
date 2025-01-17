
class LittleNeuErrors(Exception , BaseException):
    pass

class NeuralNetworksErrors(LittleNeuErrors):
    pass

class ActivationErrors(NeuralNetworksErrors):
    pass

class LossErrors(NeuralNetworksErrors):
    pass

class InternalErrors(LittleNeuErrors):
    pass

class NotImplementedErrors(LittleNeuErrors):
    pass