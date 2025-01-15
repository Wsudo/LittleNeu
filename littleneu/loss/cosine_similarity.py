import numpy

from .loss_function import LossFunction

class CosineSimilarityLoss(LossFunction):

    def __init__(self):
        super().__init__()
    
    def cosine_similarity(
            self,
            x: numpy.ndarray|list,
            y: numpy.ndarray|list,
    ) -> numpy.ndarray:
        """Compute the cosine similarity between two vectors x and y.
        Cosine similarity ranges from -1 (completely dissimilar) to 1 (completely similar).

        Args:
            x (numpy.ndarray | list): x value
            y (numpy.ndarray | list): y value

        Returns:
            numpy.ndarray: the cosine_similarity of (x ,y)
        """
        real_x = x if isinstance(x , numpy.ndarray) else numpy.array(x)
        real_y = y if isinstance(y , numpy.ndarray) else numpy.array(y)

        dot_product = numpy.dot(real_x, real_y)  # Dot product of x and y
        norm_x = numpy.linalg.norm(real_x)  # L2 norm (magnitude) of x
        norm_y = numpy.linalg.norm(real_y)  # L2 norm (magnitude) of y
        
        return dot_product / (norm_x * norm_y)

    def calc(
            self,
            y_true: numpy.ndarray|list,
            y_pred: numpy.ndarray|list,
    ) -> numpy.ndarray:
        pass