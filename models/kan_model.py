import torch
import torch.nn as nn
from kan import *

class KANWrapper(nn.Module):
    def __init__(self, width, grid=20, k=3, seed=0, bias=True, base_activation=None):
        super(KANWrapper, self).__init__()
        self.model = KAN(
            width=width,
            grid=grid,
            k=k,
            seed=seed,
        )

    def forward(self, x):
        return self.model(x)