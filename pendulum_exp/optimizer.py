"""Setup optimizer."""
from torch.optim import SGD, RMSprop

def setup_optimizer(
        params,
        opt_name: str,
        lr: float,
        dt: float,
        inverse_gradient_magnitude: float,
        weight_decay: float):
    if opt_name == 'rmsprop':
        return RMSprop(params, lr * dt, alpha=1 - dt, weight_decay=weight_decay)
    elif opt_name == 'sgd':
        return SGD(params, lr * dt * inverse_gradient_magnitude, weight_decay=weight_decay)
