"""Utilities."""
import torch
import numpy as np
from abstract import ParametricFunction

def gradient_norm(model: ParametricFunction):
    """ return norm of the gradient. """
    grad = 0
    with torch.no_grad():
        for p in model.parameters():
            grad += (p.grad ** 2).sum().item()
    return np.sqrt(grad)
