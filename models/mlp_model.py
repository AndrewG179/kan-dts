import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_len, output_len):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_len, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_len)
        )

    def forward(self, x):
        return self.net(x)