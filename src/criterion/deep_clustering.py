import torch
import torch.nn as nn
import torch.nn.functional as F

class AffinityLoss(nn.Module):
    def __init__(self, ):
        super().__init__()
        
        self.maximize = False
        
    def forward(self, input, target, batch_mean=True):
        """
        Args:
            input (batch_size, embedded_dims, n_bins, n_frames)
            target (batch_size, n_sources, n_bins, n_frames)
            
        Returns:
            loss () or (batch_size,)
        """
        batch_size, embedded_dims, n_bins, n_frames = input.size()
        batch_size, n_sources, n_bins, n_frames = target.size()

        input = input.view(batch_size, embedded_dims, n_bins*n_frames)
        target = target.view(batch_size, n_sources, n_bins*n_frames)
        input_transposed = input.permute(0,2,1).contiguous() # (batch_size, n_bins*n_frames, embedded_dims)
        target_transposed = target.permute(0,2,1).contiguous() # (batch_size, n_bins*n_frames, n_sources)

        
        affinity_input = torch.bmm(input, input_transposed) # (batch_size, embedded_dims, embedded_dims)
        affinity_target = torch.bmm(target, target_transposed) # (batch_size, n_sources, n_sources)
        affinity_correlation = torch.bmm(input, target_transposed) # (batch_size, embedded_dims, n_sources)

        loss_input = torch.sum(affinity_input**2, dim=(1,2))
        loss_target = torch.sum(affinity_target**2, dim=(1,2))
        loss_correlation = torch.sum(affinity_correlation**2, dim=(1,2))
        loss = loss_input + loss_target - 2 * loss_correlation # (batch_size,)
        
        if batch_mean:
            loss = loss.mean(dim=0) # ()
        
        return loss