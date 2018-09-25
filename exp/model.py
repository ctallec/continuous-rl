""" Define model. """
import torch.nn as nn

class Model(nn.Module):
    """ MLP. """
    def __init__(self, nb_hidden, nb_input):
        super().__init__()
        nb_layers = 2
        self.core = nn.Sequential(*([
            nn.Linear(nb_input, nb_hidden),
            nn.ReLU()
        ] + [
            nn.Linear(nb_hidden, nb_hidden),
            nn.ReLU()
        ] * nb_layers + [
            nn.Linear(nb_hidden, 1)
        ]))

    def forward(self, *inputs):
        return self.core(*inputs)
