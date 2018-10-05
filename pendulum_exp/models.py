""" Define pytorch models. """
import torch
import torch.nn as nn # pylint: disable=useless-import-alias
from noise import ParameterNoise

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

def perturbed_output(inputs: torch.Tensor, network: nn.Module, parameter_noise: ParameterNoise):
    """ Returns perturbed output. """
    with torch.no_grad():
        for p, p_noise in zip(network.parameters(), parameter_noise):
            p.copy_(p.data + p_noise)
        perturbed_output = network(inputs)
        for p, p_noise in zip(network.parameters(), parameter_noise):
            p.copy_(p.data - p_noise)
        return perturbed_output
