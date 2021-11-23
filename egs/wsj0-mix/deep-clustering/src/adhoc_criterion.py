import torch.nn as nn

EPS = 1e-12

class AffinityLossWrapper(nn.Module):
    def __init__(self, criterion):
        super().__init__()

        self.criterion = criterion
        
    def forward(self, input, target, batch_mean=True):
        """
        Args:
            input (batch_size, n_bins, n_frames, embed_dim)
            target (batch_size, n_sources, n_bins, n_frames)
        Returns:
            loss () or (batch_size,)
        """
        batch_size, n_bins, n_frames, embed_dim = input.size()
        batch_size, n_sources, n_bins, n_frames = target.size()

        input = input.view(batch_size, n_bins * n_frames, embed_dim)
        target = target.view(batch_size, n_sources, n_bins * n_frames).permute(0, 2, 1)

        loss = self.criterion(input, target, batch_mean=batch_mean)
        
        return loss