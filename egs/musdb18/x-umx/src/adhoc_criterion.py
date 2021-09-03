import torch.nn as nn

from criterion.distance import SquaredError
from criterion.sdr import NegWeightedSDR

EPS = 1e-12

class MultiDomainLoss(nn.Module):
    def  __init__(self, weight_time=10, weight_frequency=1):
        super().__init__()

        self.criterion_time = NegWeightedSDR(reduction='sum')
        self.criterion_frequency = SquaredError(reduction='sum')

        self.weight_time, self.weight_frequency = weight_time, weight_frequency
    
    def forward(self, mixture, input, target, batch_mean=True):
        weight_time, weight_frequency = self.weight_time, self.weight_frequency

        loss_time = self.criterion_time(mixture, input, target, batch_mean=batch_mean)
        loss_frequency = self.criterion_frequency(input, target, batch_mean=batch_mean)

        loss = weight_time * loss_time + weight_frequency * loss_frequency

        return loss