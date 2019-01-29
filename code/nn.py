"""Some nn utilities."""
import torch
from abstract import ParametricFunction

def copy_buffer(net: ParametricFunction, target_net: ParametricFunction):
    with torch.no_grad():
        for target_buf, buf in zip(target_net.buffers(), net.buffers()): # type: ignore
            target_buf.copy_(buf)

def soft_update(net: ParametricFunction, target_net: ParametricFunction, tau: float):
    copy_buffer(net, target_net)
    with torch.no_grad():
        for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.add_(1 - tau, param - target_param)

def hard_update(net: ParametricFunction, target_net: ParametricFunction):
    copy_buffer(net, target_net)
    with torch.no_grad():
        for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.copy_(param)
