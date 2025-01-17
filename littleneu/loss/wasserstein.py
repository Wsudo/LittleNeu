import numpy

from .loss_function import GenerativeLossFunction

from ..errors import LossErrors

class WassersteinLoss(GenerativeLossFunction):
    """Wasserstein Loss (WGAN) is a modification of the original GAN loss that aims to address some issues such as the vanishing gradient problem. Instead of using the standard binary cross-entropy loss, Wasserstein GAN uses the Wasserstein distance (also known as Earth Mover's Distance) between the real and generated data distributions.
    Args:
        y_true (numpy.ndarray | list): true values
        y_pred (numpy.ndarray | list): network generated values
        is_generator (bool) : check loss in the generator model

    Raises:
        LossErrors: when y_true is not type (list , numpy.ndarray)
        LossErrors: when y_pred is not type (list , numpy.ndarray)

    Returns:
        numpy.ndarray: loss between `True Values` and `Predicted Values`
    """

    def __init__(
        self,
        is_generator: bool = True,
    ):
        super().__init__()
        self.is_generator = is_generator

    def calc(
        self,
        y_true: numpy.ndarray | list,
        y_pred: numpy.ndarray | list,
    ) -> numpy.ndarray:
        """Wasserstein Loss (WGAN) is a modification of the original GAN loss that aims to address some issues such as the vanishing gradient problem. Instead of using the standard binary cross-entropy loss, Wasserstein GAN uses the Wasserstein distance (also known as Earth Mover's Distance) between the real and generated data distributions.
        Args:
            y_true (numpy.ndarray | list): true values
            y_pred (numpy.ndarray | list): network generated values
            is_generator (bool) : check loss in the generator model

        Raises:
            LossErrors: when y_true is not type (list , numpy.ndarray)
            LossErrors: when y_pred is not type (list , numpy.ndarray)

        Returns:
            numpy.ndarray: loss between `True Values` and `Predicted Values`
        """
        if not isinstance(y_true, numpy.ndarray) and not isinstance(y_true, list):
            raise LossErrors(f"{type(self)}.calc 'y_true' argument must be type (list , numpy.ndarray) , {type(y_true)} passed !")

        real_y_true = y_true if isinstance(y_true, numpy.ndarray) else numpy.array(y_true)
        # real_y_pred = y_pred if isinstance(y_pred, numpy.ndarray) else numpy.array(y_pred)

        # Clip predictions to avoid log(0) which leads to NaN or infinity
        real_y_pred = numpy.clip(real_y_pred, 1e-15, 1 - 1e-15)

        if self.is_generator:
            # Generator loss: Minimize - D(G(z)) to fool the discriminator
            loss = -numpy.mean(real_y_pred)  # real_y_pred here is D(G(z)), we want it to be as large as possible
        else:
            # Discriminator loss: Maximize D(x) - D(G(z)) to distinguish real and fake
            loss = numpy.mean(real_y_pred)  # real_y_pred here is D(x) for real data, we want it to be as large as possible
        return loss


class WGANLoss(WassersteinLoss):
    """Wasserstein Loss (WGAN) is a modification of the original GAN loss that aims to address some issues such as the vanishing gradient problem. Instead of using the standard binary cross-entropy loss, Wasserstein GAN uses the Wasserstein distance (also known as Earth Mover's Distance) between the real and generated data distributions.
    Args:
        y_true (numpy.ndarray | list): true values
        y_pred (numpy.ndarray | list): network generated values
        is_generator (bool) : check loss in the generator model

    Raises:
        LossErrors: when y_true is not type (list , numpy.ndarray)
        LossErrors: when y_pred is not type (list , numpy.ndarray)

    Returns:
        numpy.ndarray: loss between `True Values` and `Predicted Values`
    """

    def __init__(self, is_generator: bool = True):
        super().__init__(is_generator)


def adversarial_loss(
    y_true: numpy.ndarray | list = None,
    y_pred: numpy.ndarray | list = None,
    is_generator: bool = True,
) -> WassersteinLoss | numpy.ndarray:
    """Wasserstein Loss (WGAN) is a modification of the original GAN loss that aims to address some issues such as the vanishing gradient problem. Instead of using the standard binary cross-entropy loss, Wasserstein GAN uses the Wasserstein distance (also known as Earth Mover's Distance) between the real and generated data distributions.
    Args:
        y_true (numpy.ndarray | list): true values
        y_pred (numpy.ndarray | list): network generated values
        is_generator (bool) : check loss in the generator model

    Raises:
        LossErrors: when y_true is not type (list , numpy.ndarray)
        LossErrors: when y_pred is not type (list , numpy.ndarray)

    Returns:
        numpy.ndarray: loss between `True Values` and `Predicted Values`
    """
    return WassersteinLoss(is_generator=is_generator) if y_pred is None else WassersteinLoss(is_generator=is_generator).calc(y_true, y_pred)


def wgan_loss(
    y_true: numpy.ndarray | list = None,
    y_pred: numpy.ndarray | list = None,
    is_generator: bool = True,
) -> WGANLoss | numpy.ndarray:
    """Wasserstein Loss (WGAN) is a modification of the original GAN loss that aims to address some issues such as the vanishing gradient problem. Instead of using the standard binary cross-entropy loss, Wasserstein GAN uses the Wasserstein distance (also known as Earth Mover's Distance) between the real and generated data distributions.
    Args:
        y_true (numpy.ndarray | list): true values
        y_pred (numpy.ndarray | list): network generated values
        is_generator (bool) : check loss in the generator model

    Raises:
        LossErrors: when y_true is not type (list , numpy.ndarray)
        LossErrors: when y_pred is not type (list , numpy.ndarray)

    Returns:
        numpy.ndarray: loss between `True Values` and `Predicted Values`
    """
    return WGANLoss(is_generator=is_generator) if y_pred is None else WGANLoss(is_generator=is_generator).calc(y_true, y_pred)
