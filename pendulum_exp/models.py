"""Define pytorch models."""
from collections import OrderedDict
import torch
import torch.nn as nn
from abstract import ParametricFunction, Tensorable, Shape
from convert import check_tensor

class MLP(nn.Module, ParametricFunction):
    """MLP"""
    def __init__(self, nb_inputs: int, nb_outputs: int,
                 nb_layers: int, hidden_size: int) -> None:
        super().__init__()
        self._nb_inputs = nb_inputs
        self._nb_outputs = nb_outputs
        modules = [('fc0', nn.Linear(nb_inputs, hidden_size)), # type: ignore
                   ('ln0', nn.LayerNorm(hidden_size)),
                   ('relu0', nn.ReLU())]
        sub_core = [[(f'fc{i+1}', nn.Linear(hidden_size, hidden_size)),
                     (f'ln{i+1}', nn.LayerNorm(hidden_size)),
                     (f'relu{i+1}', nn.ReLU())] for i in range(nb_layers)]
        modules += [mod for mods in sub_core for mod in mods]
        modules += [(f'fc{nb_layers+1}', nn.Linear(hidden_size, nb_outputs))]
        self._core = nn.Sequential(OrderedDict(modules))

    def forward(self, *inputs: Tensorable):
        device = next(self.parameters())
        return self._core(check_tensor(inputs[0], device))

    def input_shape(self) -> Shape:
        return ((self._nb_inputs,),)

    def output_shape(self) -> Shape:
        return ((self._nb_outputs,),)

class ContinuousPolicyMLP(MLP, ParametricFunction):
    """MLP with a Tanh on top..."""
    def forward(self, *inputs: Tensorable):
        return torch.tanh(super().forward(*inputs))

class ContinuousAdvantageMLP(MLP, ParametricFunction):
    """MLP with 2 inputs, 1 output."""
    def __init__(self, nb_state_feats: int, nb_actions: int,
                 nb_layers: int, hidden_size: int) -> None:
        super().__init__(nb_state_feats + nb_actions, 1,
                         nb_layers, hidden_size)
        self._nb_state_feats = nb_state_feats
        self._nb_actions = nb_actions

    def forward(self, *inputs: Tensorable):
        device = next(self.parameters())
        return super().forward(torch.cat(
            [
                check_tensor(inputs[0], device),
                check_tensor(inputs[1], device)],
            dim=-1))

    def input_shape(self) -> Shape:
        return ((self._nb_state_feats,), (self._nb_actions,))

    def output_shape(self) -> Shape:
        return ((self._nb_outputs,),)

class CustomBN(nn.Module):
    def __init__(self, nb_feats: int, eps: float=1e-5) -> None:
        super().__init__()
        self._count = 0
        self._eps = eps
        self.register_buffer('_mean', torch.zeros(nb_feats, requires_grad=False))
        self.register_buffer('_squared_mean', torch.ones(nb_feats, requires_grad=False))

    def forward(self, *inputs: Tensorable) -> torch.Tensor:
        device = self._mean.device # type: ignore
        t_input = check_tensor(inputs[0], device)
        batch_size = t_input.size(0)
        std = torch.sqrt(self._squared_mean - self._mean ** 2 + self._eps) # type: ignore
        output = (t_input - self._mean) / std # type: ignore
        with torch.no_grad():
            self._mean = (self._mean * self._count + batch_size * t_input.mean(dim=0)) / (self._count + batch_size) # type: ignore
            self._squared_mean = (self._squared_mean * self._count + batch_size * (t_input ** 2).mean(dim=0)) / (self._count + batch_size) # type: ignore
            self._count += batch_size
        return output

class NormalizedMLP(nn.Module, ParametricFunction):
    def __init__(self, model: ParametricFunction) -> None:
        super().__init__()
        self._model = model
        self._bns = nn.ModuleList([CustomBN(*feat) for feat in self._model.input_shape()])

    def forward(self, *inputs: Tensorable):
        device = next(self.parameters())
        tens_inputs = [check_tensor(inp, device) for inp in inputs]
        return self._model(*[bn(inp) for (bn, inp) in zip(self._bns, tens_inputs)])

    def input_shape(self) -> Shape:
        return self._model.input_shape()

    def output_shape(self) -> Shape:
        return self._model.output_shape()
