import numpy

from ..errors import NotImplementedErrors

class LossFunction:
    
    def __init__(self):
        pass

    def calc(
            self,
            y_true:numpy.ndarray|list,
            y_pred:numpy.ndarray|list,
    ) -> numpy.ndarray:
        raise NotImplementedErrors(f"calc method not implemented for {type(self)} loss function.")

class RegressionLossFunction(LossFunction):
    pass

class ClassificationLossFunction(LossFunction):
    pass

class GenerativeLossFunction(LossFunction):
    pass

class MultiLabelLossFunction(LossFunction):
    pass

class ImbalancedClassificationLossFunction(LossFunction):
    pass

class MetricLearningLossFunction(LossFunction):
    pass

