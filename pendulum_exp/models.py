"""Define pytorch models."""
import torch
import torch.nn as nn
import torch.nn.functional as f
from abstract import ParametricFunction, Tensorable
from convert import check_tensor

class MLP(nn.Module, ParametricFunction):
    """MLP"""
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

class ContinuousPolicyMLP(MLP, ParametricFunction):
    """MLP with a Tanh on top..."""
    def forward(self, *inputs: Tensorable):
        return f.tanh(super().forward(inputs))

class ContinuousAdvantageMLP(MLP, ParametricFunction):
    """MLP with 2 inputs, 1 output."""
    def __init__(self, nb_state_feats: int, nb_actions: int,
                 nb_layers: int, hidden_size: int) -> None:
        super().__init__(nb_state_feats + nb_actions, 1,
                         nb_layers, hidden_size)

    def forward(self, *inputs: Tensorable):
        device = next(self.parameters())
        return super().forward(torch.concat(
            [
                check_tensor(inputs[0], device),
                check_tensor(inputs[1], device)],
            dim=-1))
