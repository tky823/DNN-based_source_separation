import torch
import torch.nn as nn
import torch.nn.functional as F

class FiLM(nn.Module):
    def __init__(self):
        super().__init__()

        self.gamma, self.beta = None, None

    def forward(self, input):
        return self.gamma * input + self.beta

class FiLM1d(FiLM):
    def __init__(self, num_features):
        super().__init__()

        self.gamma = nn.Parameter(torch.Tensor(1, num_features, 1))
        self.beta = nn.Parameter(torch.Tensor(1, num_features, 1))

class FiLM2d(FiLM):
    def __init__(self, num_features):
        super().__init__()

        self.gamma = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
        self.beta = nn.Parameter(torch.Tensor(1, num_features, 1, 1))

class FiLM3d(FiLM):
    def __init__(self, num_features):
        super().__init__()

        self.gamma = nn.Parameter(torch.Tensor(1, num_features, 1, 1, 1))
        self.beta = nn.Parameter(torch.Tensor(1, num_features, 1, 1, 1))

if __name__ == '__main__':
    model = FiLM1d()