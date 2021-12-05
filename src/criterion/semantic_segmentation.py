import torch
import torch.nn as nn

EPS = 1e-12

class CategoricalDiceLoss(nn.Module):
    def __init__(self, flatten_dim=(-1, -2), smooth=EPS):
        super().__init__()

        self.flatten_dim = flatten_dim
        self.smooth = smooth

    def forward(self, input, target, batch_mean=True):
        """
        Args:
            input: (batch_size, num_classes, height, width)
            target: (batch_size, num_classes, height, width)
        Returns:
            loss: () or (batch_size,)
        """
        flatten_dim = self.flatten_dim
        smooth = self.smooth

        numerator = 2 * torch.sum(input * target, dim=flatten_dim) + smooth
        denominator = input.sum(dim=flatten_dim) + target.sum(dim=flatten_dim) + smooth
        loss = torch.mean(1 - numerator / denominator, dim=1)

        if batch_mean:
            loss = loss.mean(dim=0)

        return loss