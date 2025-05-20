import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, layers, activation=nn.ReLU):
        """
        layers: list[int] → Full list of layer sizes, including input and output
                           e.g. [30, 64, 64, 1]
        activation: class → Activation function class (not instance), default is ReLU
        """
        super(MLP, self).__init__()

        net_layers = []
        for i in range(len(layers) - 1):
            net_layers.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                net_layers.append(activation())

        self.net = nn.Sequential(*net_layers)

    def forward(self, x):
        return self.net(x)