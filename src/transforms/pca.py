import torch
import torch.nn as nn

class PCA(nn.Module):
    def __init__(self, zero_mean=True):
        super().__init__()

        self.zero_mean = zero_mean
        self.mean = 0
        self.proj_matrix = None

    def forward(self, data):
        """
        Args:
            data: (batch_size, num_samples, num_features) or (num_samples, num_features)
        """
        n_dims = data.dim()

        assert n_dims == 2, "data is expected 2D tensor."
        
        if self.training:
            if self.zero_mean:
                mean = torch.mean(data, dim=0)
                self.mean = mean
                normalized = data - mean
            else:
                normalized = data
            
            cov = torch.matmul(normalized.permute(1, 0), normalized) / normalized.size(0)
            _, proj_matrix = torch.linalg.eigh(cov) # Computed by assending order
            self.proj_matrix = torch.flip(proj_matrix, dims=(-1,))
        else:
            if self.proj_matrix is None:
                raise RuntimeError("proj_matrix is computed in advance.")
            
            if self.zero_mean:
                normalized = data - self.mean
            else:
                normalized = data
        
        output = torch.matmul(normalized, self.proj_matrix)

        return output