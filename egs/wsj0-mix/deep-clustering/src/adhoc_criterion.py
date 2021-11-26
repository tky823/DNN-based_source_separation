import torch.nn as nn

EPS = 1e-12

class AffinityLossWrapper(nn.Module):
    def __init__(self, criterion):
        super().__init__()

        self.criterion = criterion
        
    def forward(self, input, target, binary_mask=None, batch_mean=True):
        """
        Args:
            input: (batch_size, n_bins, n_frames, embed_dim)
            target: (batch_size, n_sources, n_bins, n_frames)
            binary_mask: (batch_size, 1, n_bins, n_frames)
        Returns:
            loss: () or (batch_size,)
        """
        batch_size, n_bins, n_frames, embed_dim = input.size()
        batch_size, n_sources, n_bins, n_frames = target.size()

        input = input.view(batch_size, n_bins * n_frames, embed_dim)
        target = target.view(batch_size, n_sources, n_bins * n_frames).permute(0, 2, 1)
        if binary_mask is not None:
            binary_mask = binary_mask.view(batch_size, n_bins * n_frames)

        loss = self.criterion(input, target, binary_mask=binary_mask, batch_mean=batch_mean)
        
        return loss