from __future__ import division, print_function
import numpy as np
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, size):
        super(Model, self).__init__()
        self.size = size
        self.dnn = torch.nn.Sequential(  # 1*90
            torch.nn.Linear(size, self.size * 0.4),
            torch.nn.ReLU(),
            torch.nn.Linear(self.size * 0.4, 2),
        )

    def forward(self, x):
        x = x.view(-1, self.size)
        x = self.dnn(x)
        # print(x.shape)
        return x
