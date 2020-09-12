import itertools
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
        self.anchor = nn.Parameter(torch.Tensor(n_anchors, embed_dim), requires_grad=True)
    
    def forward(self, input, threshold_weight=None, n_sources=None):
        """
        Args:
            input (batch_size, 1, F_bin, T_bin): Amplitude
            assignment (batch_size, n_sources, F_bin, T_bin): Speaker assignment when training
            threshold_weight (batch_size, 1, F_bin, T_bin) or <float>
        Returns:
            output (batch_size, n_sources, F_bin, T_bin)
        """
        output, _ = self.extract_latent(input, threshold_weight=threshold_weight, n_sources=n_sources)
        
        return output
    
    def extract_latent(self, input, threshold_weight=None, n_sources=None):
        """
        Args:
            input (batch_size, 1, F_bin, T_bin) <torch.Tensor>
            threshold_weight (batch_size, 1, F_bin, T_bin) or <float>
        """
        if n_sources is None:
            raise ValueError("Specify n_sources!")
        
        n_anchors = self.n_anchors
        patterns = list(itertools.combinations(range(n_anchors), n_sources))
        patterns = torch.Tensor(patterns).long()
        n_patterns = len(patterns)
        diag_mask = torch.diag(torch.ones((n_sources,), dtype=torch.float))
        
        embed_dim = self.embed_dim
        eps = self.eps
        
        batch_size, _, F_bin, T_bin = input.size()
        num_features = F_bin*T_bin
        
        log_amplitude = torch.log(input + eps)
        x = self.lstm(log_amplitude) # -> (batch_size, T_bin, F_bin)
        x = self.fc(x) # -> (batch_size, T_bin, embed_dim*F_bin)
        x = x.view(batch_size, T_bin, embed_dim, F_bin)
        x = x.permute(0,2,3,1).contiguous()  # -> (batch_size, embed_dim, F_bin, T_bin)
        latent = x.view(batch_size, embed_dim, F_bin*T_bin)
        
        if self.training:
            anchor_permutation = self.anchor[patterns].unsqueeze(dim=0) # (1, n_patterns, n_sources, embed_dim)
            anchor_permutation = anchor_permutation.repeat(batch_size, 1, 1, 1) # (batch_size, n_patterns, n_sources, embed_dim)
            anchor_permutation = anchor_permutation.view(batch_size*n_patterns, n_sources, embed_dim)
            latent_patterns = latent.unsqueeze(dim=1) # (batch_size, 1, embed_dim, F_bin*T_bin)
            latent_patterns = latent_patterns.repeat(1, n_patterns, 1, 1) # (batch_size, n_patterns, embed_dim, F_bin*T_bin)
            latent_patterns = latent_patterns.view(batch_size*n_patterns, embed_dim, F_bin*T_bin)
            similarity_permutation = torch.bmm(anchor_permutation, latent_patterns) # (batch_size*n_patterns, n_sources, F_bin*T_bin)
            assignment = torch.softmax(similarity_permutation, dim=1) # -> (batch_size*n_patterns, n_sources, F_bin*T_bin)
            assignment = assignment.view(batch_size, n_patterns, n_sources, F_bin*T_bin)
            
            threshold_weight = threshold_weight.view(batch_size, 1, 1, F_bin*T_bin)
            
            assignment = threshold_weight * assignment # -> (batch_size, n_patterns, n_sources, F_bin*T_bin)
            assignment = assignment.view(batch_size*n_patterns, n_sources, F_bin*T_bin)
            attractor = torch.bmm(assignment, latent_patterns.permute(0,2,1)) / (assignment.sum(dim=2, keepdim=True) + eps) # -> (batch_size*n_patterns, n_sources, embed_dim)
        else:
            raise ValueError("Sorry, I haven't implemented...")
        
        similarity = torch.bmm(attractor, attractor.permute(0,2,1)) # -> (batch_size*n_patterns, n_sources, n_sources)
        similarity = similarity.view(batch_size, n_patterns, n_sources, n_sources)
        
        similarity = (1 - diag_mask) * similarity + (torch.min(similarity) - 1) * diag_mask * similarity
        similarity, _ = torch.max(similarity, dim=3) # -> (batch_size, n_patterns, n_sources)
        similarity, _ = torch.max(similarity, dim=2) # -> (batch_size, n_patterns)
        patterns_idx = torch.arange(0, batch_size*n_patterns, n_patterns) + torch.argmin(similarity, dim=1) # -> (batch_size,)
        attractor = attractor[patterns_idx] # -> (batch_size, n_sources, embed_dim)
        
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

if __name__ == '__main__':
    from algorithm.ideal_mask import ideal_binary_mask
    
    torch.manual_seed(111)
    
    batch_size = 2
    K = 10
    
    H = 32
    B = 4
    
    F_bin, T_bin = 4, 128
    C = 2
    causal = False
    mask_nonlinear = 'sigmoid'
    
    sources = torch.randint(0, 10, (batch_size, C, F_bin, T_bin), dtype=torch.float)
    input = sources.sum(dim=1, keepdim=True)
    threshold_weight = torch.randint(0, 2, (batch_size, 1, F_bin, T_bin), dtype=torch.float)
    
    model = ADANet(F_bin, embed_dim=K, hidden_channels=H, num_blocks=B, causal=causal, mask_nonlinear=mask_nonlinear, n_sources=C)
    print(model)
    output = model(input, threshold_weight=threshold_weight, n_sources=C)
    
    print(input.size(), output.size())
