""" Define pytorch models. """
from typing import Union
import torch
import torch.nn as nn # pylint: disable=useless-import-alias
from noise import ParameterNoise, ActionNoise

class MLP(nn.Module):
    """ MLP """
    def __init__(self, nb_inputs: int, nb_outputs: int,
                 nb_layers: int, hidden_size: int) -> None:
        super().__init__()
        modules = (
            [nn.Linear(nb_inputs, hidden_size), nn.ReLU()] +
            nb_layers * [nn.Linear(hidden_size, hidden_size), nn.ReLU()] +
            [nn.Linear(hidden_size, nb_outputs)])
        self._core = nn.Sequential(*modules)

    def forward(self, *inputs):
        return self._core(*inputs)

def perturbed_output(inputs: torch.Tensor,
                     network: nn.Module,
                     noise: Union[ParameterNoise, ActionNoise]):
    """ Returns perturbed_output. """
    if isinstance(noise, ParameterNoise):
        return params_perturbed_output(inputs, network, noise)
    return action_perturbed_output(inputs, network, noise)

def params_perturbed_output(
        inputs: torch.Tensor,
        network: nn.Module,
        parameter_noise: ParameterNoise):
    """ Returns perturbed output. """
    with torch.no_grad():
        for p, p_noise in zip(network.parameters(), parameter_noise):
            p.copy_(p.data + p_noise)
        perturbed_output = network(inputs)
        for p, p_noise in zip(network.parameters(), parameter_noise):
            p.copy_(p.data - p_noise)
        return perturbed_output

def action_perturbed_output(
        inputs: torch.Tensor,
        network: nn.Module,
        action_noise: ActionNoise):
    """ Returns perturbed output. """
    with torch.no_grad():
        return network(inputs) + action_noise.noise
