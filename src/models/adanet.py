import itertools

import torch
import torch.nn as nn

from models.danet import DANet

EPS = 1e-12

"""
    Anchored DANet
"""
class ADANet(DANet):
    def __init__(self, n_bins, embed_dim=20, hidden_channels=600, num_blocks=4, num_anchors=6, dropout=5e-1, causal=False, mask_nonlinear='sigmoid', eps=EPS, **kwargs):
        super().__init__(n_bins, embed_dim=embed_dim, hidden_channels=hidden_channels, num_blocks=num_blocks, dropout=dropout, causal=causal, mask_nonlinear=mask_nonlinear, eps=eps, **kwargs)
        
        self.num_anchors = num_anchors
        self.anchor = nn.Parameter(torch.Tensor(num_anchors, embed_dim), requires_grad=True)
    
    def forward(self, input, threshold_weight=None, n_sources=None):
        """
        Args:
            input <torch.Tensor>: Amplitude tensor with shape of (batch_size, 1, n_bins, n_frames)
            threshold_weight <torch.Tensor> or <float>: Tensor shape of (batch_size, 1, n_bins, n_frames).
        Returns:
            output <torch.Tensor>: (batch_size, n_sources, n_bins, n_frames)
        """
        output, _ = self.extract_latent(input, threshold_weight=threshold_weight, n_sources=n_sources)
        
        return output
    
    def extract_latent(self, input, threshold_weight=None, n_sources=None):
        """
        Args:
            input <torch.Tensor>: Amplitude tensor with shape of (batch_size, 1, n_bins, n_frames)
            threshold_weight <torch.Tensor> or <float>: Tensor shape of (batch_size, 1, n_bins, n_frames).
            n_sources <int>: Number of sources in mixture.
        """
        if n_sources is None:
            raise ValueError("Specify n_sources!")
        
        num_anchors = self.num_anchors
        embed_dim = self.embed_dim
        eps = self.eps

        diag_mask = torch.eye(n_sources)
        diag_mask = diag_mask.to(input.device) # (n_sources, n_sources)

        patterns = list(itertools.combinations(range(num_anchors), n_sources))
        n_patterns = len(patterns)
        patterns = torch.Tensor(patterns).long()
        patterns = patterns.to(self.anchor.device)
        anchor_combination = self.anchor[patterns] # (n_patterns, n_sources, embed_dim)
        
        batch_size, _, n_bins, n_frames = input.size()

        self.rnn.flatten_parameters()
        
        log_amplitude = torch.log(input + eps)
        x = log_amplitude.squeeze(dim=1).permute(0, 2, 1).contiguous() # (batch_size, n_frames, n_bins)
        x, _ = self.rnn(x) # (batch_size, n_frames, n_bins)
        x = self.fc(x) # (batch_size, n_frames, embed_dim * n_bins)
        x = x.view(batch_size, n_frames, embed_dim, n_bins)
        x = x.permute(0, 2, 3, 1).contiguous()  # (batch_size, embed_dim, n_bins, n_frames)
        latent = x.view(batch_size, embed_dim, n_bins * n_frames)
        latent = latent.permute(0, 2, 1).contiguous() # (batch_size, n_bins * n_frames, embed_dim)

        distance_combination = []

        for anchor in anchor_combination:
            distance = torch.sum(anchor.unsqueeze(dim=1) * latent.unsqueeze(dim=1), dim=-1) # (batch_size, n_sources, n_bins * n_frames)
            distance_combination.append(distance)
        
        distance_combination = torch.stack(distance_combination, dim=0) # (n_patterns, batch_size, n_sources, n_bins * n_frames)
        assignment_combination = torch.softmax(distance_combination, dim=2) # (n_patterns, batch_size, n_sources, n_bins * n_frames)

        if threshold_weight is not None:
            threshold_weight = threshold_weight.view(1, batch_size, 1, n_bins * n_frames)
            assignment_combination = threshold_weight * assignment_combination # (n_patterns, batch_size, n_sources, n_bins * n_frames)

        attractor_combination, max_similarity_combination = [], []

        for assignment in assignment_combination:
            attractor = torch.bmm(assignment, latent) / (assignment.sum(dim=2, keepdim=True) + eps) # (batch_size, n_sources, embed_dim)
            similarity = torch.bmm(attractor, attractor.permute(0, 2, 1)) # (batch_size, n_sources, n_sources)
            masked_similarity = diag_mask * similarity
            masked_similarity = masked_similarity.view(batch_size, n_sources * n_sources) # (batch_size, n_sources * n_sources)
            max_similarity, _ = torch.max(masked_similarity, dim=1) # (batch_size,)

            attractor_combination.append(attractor)
            max_similarity_combination.append(max_similarity)
        
        attractor_combination = torch.stack(attractor_combination, dim=1) # (batch_size, n_patterns, n_sources, embed_dim)
        flatten_attractor_combination = attractor_combination.view(batch_size * n_patterns, n_sources, embed_dim)
        max_similarity_combination = torch.stack(max_similarity_combination, dim=1) # (batch_size, n_patterns)
        indices = torch.argmin(max_similarity_combination, dim=1) # (batch_size,)
        flatten_indices = indices + torch.arange(0, batch_size * n_patterns, n_patterns) # (batch_size,)
        flatten_indices = flatten_indices.long()
        flatten_indices = flatten_indices.to(flatten_attractor_combination.device)
        attractor = flatten_attractor_combination[flatten_indices] # (batch_size, n_sources, embed_dim)

        similarity = torch.bmm(attractor, latent.permute(0, 2, 1)) # (batch_size, n_sources, n_bins * n_frames)
        similarity = similarity.view(batch_size, n_sources, n_bins, n_frames)
        mask = self.mask_nonlinear2d(similarity) # (batch_size, n_sources, n_bins, n_frames)
        output = mask * input

        return output, latent
    
    def _reset_parameters(self):
        nn.init.orthogonal_(self.anchor.data)
    
    def get_config(self):
        config = super().get_config()
        config['num_anchors'] = self.num_anchors
        
        return config
    
    @classmethod
    def build_model(cls, model_path, load_state_dict=False):
        config = torch.load(model_path, map_location=lambda storage, loc: storage)
        
        n_bins = config['n_bins']
        embed_dim = config['embed_dim']
        hidden_channels = config['hidden_channels']
        num_blocks = config['num_blocks']
        num_anchors = config['num_anchors']
        dropout = config['dropout']
        
        causal = config['causal']
        mask_nonlinear = config['mask_nonlinear']
        iter_clustering = config['iter_clustering']
        
        eps = config['eps']
        
        model = cls(n_bins, embed_dim=embed_dim, hidden_channels=hidden_channels, num_blocks=num_blocks, num_anchors=num_anchors, dropout=dropout, causal=causal, mask_nonlinear=mask_nonlinear, iter_clustering=iter_clustering, eps=eps)

        if load_state_dict:
            model.load_state_dict(config['state_dict'])
        
        return model
    
    @property
    def num_parameters(self):
        _num_parameters = 0
        
        for p in self.parameters():
            if p.requires_grad:
                _num_parameters += p.numel()
                
        return _num_parameters

def _test_adanet():
    torch.manual_seed(111)

    batch_size = 2
    N = 6
    K = 10
    H = 32
    B = 4
    
    n_bins, n_frames = 4, 128
    n_sources = 2
    causal = False
    mask_nonlinear = 'sigmoid'
    
    sources = torch.randn((batch_size, n_sources, n_bins, n_frames), dtype=torch.float)
    input = sources.sum(dim=1, keepdim=True)
    threshold_weight = torch.randint(0, 2, (batch_size, 1, n_bins, n_frames), dtype=torch.float)
    
    model = ADANet(n_bins, embed_dim=K, hidden_channels=H, num_blocks=B, num_anchors=N, causal=causal, mask_nonlinear=mask_nonlinear)
    print(model)
    print("# Parameters: {}".format(model.num_parameters))

    output = model(input, threshold_weight=threshold_weight, n_sources=n_sources)
    
    print(input.size(), output.size())

def _test_adanet_paper():
    batch_size = 2
    N = 6
    K = 20
    H = 300
    B = 4
    
    n_bins, n_frames = 129, 256
    n_sources = 2
    causal = False
    mask_nonlinear = 'sigmoid'
    
    sources = torch.randn((batch_size, n_sources, n_bins, n_frames), dtype=torch.float)
    input = sources.sum(dim=1, keepdim=True)
    threshold_weight = torch.randint(0, 2, (batch_size, 1, n_bins, n_frames), dtype=torch.float)
    
    model = ADANet(n_bins, embed_dim=K, hidden_channels=H, num_blocks=B, num_anchors=N, causal=causal, mask_nonlinear=mask_nonlinear)
    print(model)
    print("# Parameters: {}".format(model.num_parameters))

    output = model(input, threshold_weight=threshold_weight, n_sources=n_sources)
    
    print(input.size(), output.size())

if __name__ == '__main__':
    torch.manual_seed(111)

    print("="*10, "ADANet", "="*10)
    _test_adanet()
    print()

    print("="*10, "ADANet (paper)", "="*10)
    _test_adanet_paper()
