"""Setup optimizer."""
from typing import Optional
from logging import info
from torch.optim import SGD, RMSprop, Adam

def setup_optimizer(
        params, opt_name: str, lr: float,
        dt: float, inverse_gradient_magnitude: Optional[float] = None,
        weight_decay: Optional[float] = None):
    """Setup optimizer, and scale optimizer parameters according to the framerate dt.

    :args params: iterable on the torch.Parameters to optimize
    :args opt_name: one of "rmsprop", "sgd"
    :args lr: lr before scaling
    :args dt: framerate
    :args inverse_gradient_magnitude: deprecated
    :args weight_decay: weight decay used

    :return: corresponding torch.Optimizer
    """
    if opt_name == 'rmsprop':
        info(f"setup> using RMSprop, the provided lr {lr} is scaled, actual values: "
             f"lr={lr * dt}, alpha={1 - dt}")
        return RMSprop(params, lr * dt, alpha=1 - dt, weight_decay=weight_decay)
    elif opt_name == 'sgd':
        assert inverse_gradient_magnitude is not None
        info(f"setup> !!!DEPRECATED!!! using SGD, the provided lr {lr} is scaled, actual values: "
             f"lr={lr * dt * inverse_gradient_magnitude}")
        return SGD(params, lr * dt * inverse_gradient_magnitude, weight_decay=weight_decay)
    elif opt_name == 'adam':
        return Adam(params)
