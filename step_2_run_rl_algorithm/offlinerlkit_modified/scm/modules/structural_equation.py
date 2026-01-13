"""Generic structural equation class without specified model class/archtitecture: Variational autoencoders or normalizing flows."""

from torch import nn

class StructuralEquation(nn.Module):
    def __init__(self, var_dim=2):
        super().__init__()
        self.latent_dim = var_dim # the dimension of variables

    def encode(self):
        raise NotImplementedError

    def decode(self):
        raise NotImplementedError