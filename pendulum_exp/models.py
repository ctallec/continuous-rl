"""Define pytorch models."""
import torch.nn as nn
from abstract import ParametricFunction, Tensorable
from convert import check_tensor

class MLP(nn.Module, ParametricFunction):
    """ MLP """
    def __init__(self, nb_inputs: int, nb_outputs: int,
                 nb_layers: int, hidden_size: int) -> None:
        super().__init__()
        modules = (
            [nn.Linear(nb_inputs, hidden_size), nn.ReLU()] +
            nb_layers * [nn.Linear(hidden_size, hidden_size), nn.ReLU()] +
            [nn.Linear(hidden_size, nb_outputs)])
        self._core = nn.Sequential(*modules)

    def forward(self, *inputs: Tensorable):
        device = next(self.parameters())
        return self._core(check_tensor(inputs[0], device))
