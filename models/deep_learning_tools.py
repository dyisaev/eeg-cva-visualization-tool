import torch
from torch import nn 
class MLP_2HiddenLayers(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP_2HiddenLayers, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 14),
            nn.ReLU(),
        )
        self.lastlayer= nn.Linear(14, output_dim)

    def forward(self, x):

        logits = self.lastlayer(self.linear_relu_stack(x))

        return logits