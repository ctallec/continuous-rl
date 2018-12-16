"""Define pytorch models."""
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as f
from abstract import ParametricFunction, Tensorable, Shape, Arrayable
from convert import check_tensor, check_array, arr_to_th
# from mylog import log
# from uuid import uuid4

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
                 nb_layers: int, hidden_size: int, nb_outputs: int) -> None:
        super().__init__(nb_state_feats + nb_actions, nb_outputs,
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
    def __init__(self, nb_feats: int, eps: float = 1e-5) -> None:
        super().__init__()
        self._eps = eps
        self.register_buffer('_count', torch.zeros(1, requires_grad=False))
        self.register_buffer('_mean', torch.zeros(nb_feats, requires_grad=False))
        self.register_buffer('_squared_mean', torch.ones(nb_feats, requires_grad=False))

        # debug: we are going to log _count, _mean and _squared_mean
        # self._prefix = 'stats/' + str(uuid4())

    def forward(self, *inputs: Tensorable) -> torch.Tensor:
        device = self._mean.device # type: ignore
        t_input = check_tensor(inputs[0], device)
        batch_size = t_input.size(0)

        # log
        # count = int(self._count.item())
        # if (count // batch_size) % 100 == 99:
        #     log(self._prefix + 'count', count, count)
        #     log(self._prefix + 'min_mean', self._mean.abs().min(), count) # type: ignore
        #     log(self._prefix + 'max_mean', self._mean.abs().max(), count) # type: ignore
        #     log(self._prefix + 'min_sq_mean', self._squared_mean.min(), count) # type: ignore
        #     log(self._prefix + 'max_sq_mean', self._squared_mean.max(), count) # type: ignore
        std = torch.sqrt(torch.clamp(self._squared_mean - self._mean ** 2, min=1e-2)) # type: ignore
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
        # only normalize first input (is this what we want to do in the long go?)
        self._bn = CustomBN(self._model.input_shape()[0][0])

    def forward(self, *inputs: Tensorable):
        device = next(self.parameters())
        tens_inputs = [check_tensor(inp, device) for inp in inputs]
        tens_inputs = [self._bn(tens_inputs[0])] + tens_inputs[1:]
        return self._model(*tens_inputs)

    def input_shape(self) -> Shape:
        return self._model.input_shape()

    def output_shape(self) -> Shape:
        return self._model.output_shape()

class MixtureNetwork(nn.Module, ParametricFunction):
    def __init__(self, value_net: ParametricFunction,
                 mixture_net: ParametricFunction,
                 val_scale: Arrayable, logpi_scale: Arrayable) -> None:
        super().__init__()
        self._value_net = value_net
        self._mixture_net = mixture_net
        self._val_scale = check_array([[val_scale]])
        self._logpi_scale = check_array([[logpi_scale]])

    def forward(self, *inputs: Tensorable):
        device = next(self.parameters())
        tens_inputs = [check_tensor(inp) for inp in inputs]
        batch_size = tens_inputs[0].size(0)
        val = self._value_net(*tens_inputs)
        logpi = self._mixture_net(*tens_inputs)
        nb_mix = self._val_scale.shape[-1]
        val = val.view(batch_size, nb_mix, -1) * arr_to_th(self._val_scale, device)
        logpi = f.log_softmax(val.view(batch_size, nb_mix, -1) + arr_to_th(self._logpi_scale, device), dim=-2)
        return (val * logpi.exp()).sum(dim=-2)

    def input_shape(self) -> Shape:
        return self._value_net.input_shape()

    def output_shape(self) -> Shape:
        o_s, = self._value_net.output_shape()
        return (o_s // self._val_scale.shape[-1],)
