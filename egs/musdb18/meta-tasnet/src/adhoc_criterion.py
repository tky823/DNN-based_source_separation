import torch.nn as nn

class MultiLoss(nn.Module):
    def  __init__(self, metrics, weights):
        super().__init__()

        self.metrics = nn.ModuleDict(metrics)
        self.weights = weights
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError
