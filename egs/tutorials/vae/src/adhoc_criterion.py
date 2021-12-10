import torch
import torch.nn as nn

EPS = 1e-12

class KLdivergence(nn.Module):
    def __init__(self, eps=EPS):
        super().__init__()

        self.eps = eps
        
    def forward(self, mean, var, batch_mean=True):
        """
        Args:
            mean (batch_size, latent_dim)
            var (batch_size, latent_dim)
        """
        eps = self.eps

        loss = 1 + torch.log(var + eps) - mean**2 - var
        loss = - 0.5 * loss.sum(dim=1)

        if batch_mean:
            loss = loss.mean(dim=0)

        return loss