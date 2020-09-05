import torch
import torch.nn as nn

from models.danet import DANet

EPS=1e-12

"""
    Anchored DANet
    
"""

class ADANet(DANet):
    def __init__(self, F_bin, embed_dim=20, hidden_channels=600, num_blocks=4, causal=False, n_anchors=6, mask_nonlinear='sigmoid', iter_clustering=10, eps=EPS, **kwargs):
        super().__init__(F_bin, embed_dim=embed_dim, hidden_channels=hidden_channels, num_blocks=num_blocks, causal=causal, mask_nonlinear=mask_nonlinear, iter_clustering=iter_clustering, eps=eps, **kwargs)
        
        self.n_anchors = n_anchors
    
    def extract_latent(self, input, assignment=None, threshold_weight=None, n_sources=None, iter_clustering=None):
        """
        input (batch_size, 1, F_bin, T_bin) <torch.Tensor>
        assignment (batch_size, n_sources, F_bin, T_bin) <torch.Tensor>
        threshold_weight (batch_size, 1, F_bin, T_bin) or <float>
        """
        if iter_clustering is None:
            iter_clustering = self.iter_clustering
        
        embed_dim = self.embed_dim
        
        if n_sources is not None:
            if assignment is not None and n_sources != assignment.size(1):
                raise ValueError("n_sources is different from assignment.size(1)")
        else:
            if assignment is None:
                raise ValueError("Specify assignment, given None!")
            n_sources = assignment.size(1)
        
        eps = self.eps
        
        batch_size, _, F_bin, T_bin = input.size()
        
        log_amplitude = torch.log(input + eps)
        x = self.lstm(log_amplitude) # -> (batch_size, T_bin, F_bin)
        x = self.fc(x) # -> (batch_size, T_bin, embed_dim*F_bin)
        x = x.view(batch_size, T_bin, embed_dim, F_bin)
        x = x.permute(0,2,3,1).contiguous()  # -> (batch_size, embed_dim, F_bin, T_bin)
        latent = x.view(batch_size, embed_dim, F_bin*T_bin)
        
        if assignment is None:
            # TODO: test threshold
            if self.training:
                raise ValueError("assignment is required.")
            latent_kmeans = latent.squeeze(dim=0) # -> (embed_dim, F_bin*T_bin)
            latent_kmeans = latent_kmeans.permute(1,0) # -> (F_bin*T_bin, embed_dim)
            kmeans = Kmeans(latent_kmeans, K=n_sources)
            _, centroids = kmeans(iteration=iter_clustering) # (F_bin*T_bin, n_sources), (n_sources, embed_dim)
            attractor = centroids.unsqueeze(dim=0) # (batch_size, n_sources, embed_dim)
        else:
            threshold_weight = threshold_weight.view(batch_size, 1, F_bin*T_bin)
            assignment = assignment.view(batch_size, n_sources, F_bin*T_bin) # -> (batch_size, n_sources, F_bin*T_bin)
            assignment = threshold_weight * assignment
            attractor = torch.bmm(assignment, latent.permute(0,2,1)) / (assignment.sum(dim=2, keepdim=True) + eps) # -> (batch_size, n_sources, embed_dim)
        
        similarity = torch.bmm(attractor, latent) # -> (batch_size, n_sources, F_bin*T_bin)
        similarity = similarity.view(batch_size, n_sources, F_bin, T_bin)
        mask = self.mask_nonlinear2d(similarity) # -> (batch_size, n_sources, F_bin, T_bin)
        output = mask * input
        
        return output, latent
    
    def _get_num_parameters(self):
        num_parameters = 0
        
        for p in self.parameters():
            if p.requires_grad:
                num_parameters += p.numel()
                
        return num_parameters
