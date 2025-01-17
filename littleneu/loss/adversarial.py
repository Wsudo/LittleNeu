import numpy

from .loss_function import GenerativeLossFunction

from ..errors import LossErrors

class AdversarialLoss(GenerativeLossFunction):
    """In Generative Adversarial Networks (GANs), the loss is derived from a min-max game between two networks: a generator (G) and a discriminator (D). The adversarial loss is designed to measure how well the generator is able to create data that mimics real data, and how well the discriminator distinguishes between real and fake data.

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
        """In Generative Adversarial Networks (GANs), the loss is derived from a min-max game between two networks: a generator (G) and a discriminator (D). The adversarial loss is designed to measure how well the generator is able to create data that mimics real data, and how well the discriminator distinguishes between real and fake data.

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

        if self.is_ge:
            # Generator loss: G tries to fool the discriminator, we want D(G(z)) to be close to 1
            # Generator tries to maximize log(D(G(z)))
            loss = -numpy.mean(numpy.log(real_y_pred))  # real_y_pred here is D(G(z)), aiming for 1 (real)
        else:
            # Discriminator loss: D tries to classify real and fake correctly
            # Discriminator wants to maximize log(D(x)) for real data and log(1 - D(G(z))) for fake data
            real_loss = -numpy.mean(numpy.log(real_y_pred))  # Real data, y_true would be 1
            fake_loss = -numpy.mean(numpy.log(1 - real_y_pred))  # Fake data, y_true would be 0
            loss = real_loss + fake_loss

        return loss


class AGANLoss(AdversarialLoss):
    """In Generative Adversarial Networks (GANs), the loss is derived from a min-max game between two networks: a generator (G) and a discriminator (D). The adversarial loss is designed to measure how well the generator is able to create data that mimics real data, and how well the discriminator distinguishes between real and fake data.

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
) -> AdversarialLoss | numpy.ndarray:
    """In Generative Adversarial Networks (GANs), the loss is derived from a min-max game between two networks: a generator (G) and a discriminator (D). The adversarial loss is designed to measure how well the generator is able to create data that mimics real data, and how well the discriminator distinguishes between real and fake data.

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
    return AdversarialLoss(is_generator=is_generator) if y_pred is None else AdversarialLoss(is_generator=is_generator).calc(y_true, y_pred)


def agan_loss(
    y_true: numpy.ndarray | list = None,
    y_pred: numpy.ndarray | list = None,
    is_generator: bool = True,
) -> AGANLoss | numpy.ndarray:
    """In Generative Adversarial Networks (GANs), the loss is derived from a min-max game between two networks: a generator (G) and a discriminator (D). The adversarial loss is designed to measure how well the generator is able to create data that mimics real data, and how well the discriminator distinguishes between real and fake data.

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
    return AGANLoss(is_generator=is_generator) if y_pred is None else AGANLoss(is_generator=is_generator).calc(y_true, y_pred)
