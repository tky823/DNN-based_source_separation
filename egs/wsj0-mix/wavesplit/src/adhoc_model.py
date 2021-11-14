import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.wavesplit import WaveSplitBase

EPS = 1e-12

class WaveSplit(WaveSplitBase):
    def __init__(self, speaker_stack: nn.Module, separation_stack: nn.Module, latent_dim: int, n_sources=2, n_training_sources=10, spk_criterion=None, eps=EPS):
        super().__init__(speaker_stack, separation_stack, n_sources=n_sources, n_training_sources=n_training_sources, spk_criterion=spk_criterion)
    
        self.embedding = nn.Embedding(n_training_sources, latent_dim)

        self.eps = eps

    def forward(self, mixture, spk_idx=None, sorted_idx=None, return_all_layers=False, return_spk_vector=False, return_spk_embedding=False, return_all_spk_embedding=False, stack_dim=1):
        """
        Args:
            mixture: (batch_size, 1, T)
            spk_idx: (batch_size, n_sources)
            sorted_idx: (batch_size, n_sources)
        Returns:

        """
        if self.training:
            output = self.training_forward(mixture, spk_idx=spk_idx, sorted_idx=sorted_idx, return_all_layers=return_all_layers, return_spk_vector=return_spk_vector, return_spk_embedding=return_spk_embedding, return_all_spk_embedding=return_all_spk_embedding, stack_dim=stack_dim)
        else:
            if spk_idx is not None or sorted_idx is not None or return_spk_embedding:
                raise NotImplementedError
            
            output = self.evaluation_forward(mixture, return_all_layers=return_all_layers, return_spk_vector=return_spk_vector, return_all_spk_embedding=return_all_spk_embedding, stack_dim=stack_dim)

        return output
    
    def training_forward(self, mixture, spk_idx=None, sorted_idx=None, return_all_layers=False, return_spk_vector=False, return_spk_embedding=False, return_all_spk_embedding=False, stack_dim=1):
        """
        Args:
            mixture: (batch_size, 1, T)
            spk_idx: (batch_size, n_sources)
            sorted_idx: (batch_size, n_sources)
        Returns:

        """
        n_sources = self.n_sources
        eps = self.eps

        spk_vector = self.speaker_stack(mixture) # (batch_size, n_sources, latent_dim, T)
        
        all_spk_embedding = self.embedding(self.all_spk_idx) # (n_training_sources, latent_dim)
        all_spk_embedding = all_spk_embedding / (torch.linalg.vector_norm(all_spk_embedding, dim=1, keepdim=True) + eps)
        spk_embedding = all_spk_embedding[spk_idx] # (batch_size, n_sources, latent_dim)

        if sorted_idx is None:
            if return_all_layers or return_spk_vector or return_spk_embedding or return_all_spk_embedding:
                raise ValueError("Set return_all_layers=False, return_spk_vector=False, return_spk_embedding=False, return_all_spk_embedding=False.")
                
            spk_vector = spk_vector.permute(0, 3, 1, 2).contiguous() # (batch_size, T, n_sources, latent_dim)
            _, sorted_idx = self.compute_pit_speaker_loss(spk_vector, spk_embedding, all_spk_embedding, feature_last=True)

            return sorted_idx
        
        # Reorder speaker vector using sorted_idx.
        batch_size, _, latent_dim, T = spk_vector.size()

        additional_idx = torch.arange(0, batch_size * n_sources, n_sources).unsqueeze(dim=1)
        sorted_idx = sorted_idx + additional_idx.to(sorted_idx.device)
        flatten_spk_vector = spk_vector.view(batch_size * n_sources, latent_dim, T)
        flatten_sorted_idx = sorted_idx.view(batch_size * n_sources)
        flatten_sorted_spk_vector = flatten_spk_vector[flatten_sorted_idx]
        sorted_spk_vector = flatten_sorted_spk_vector.view(batch_size, n_sources, latent_dim, T)
        spk_centroids = sorted_spk_vector.mean(dim=-1) # (batch_size, n_sources, latent_dim)

        estimated_sources = self.separation_stack(mixture, spk_centroids, return_all=return_all_layers, stack_dim=stack_dim)

        output = []
        output.append(estimated_sources)

        if return_spk_vector:    
            output.append(sorted_spk_vector)
        
        if return_spk_embedding:
            output.append(spk_embedding)
        
        if return_all_spk_embedding:    
            output.append(all_spk_embedding)
        
        if len(output) == 1:
            output = output[0]
        else:
            output = tuple(output)

        return output
    
    def evaluation_forward(self, mixture, return_all_layers=False, return_spk_vector=False, return_all_spk_embedding=False, stack_dim=1):
        eps = self.eps

        all_spk_embedding = self.embedding(self.all_spk_idx) # (n_training_sources, latent_dim)
        all_spk_embedding = all_spk_embedding / (torch.linalg.vector_norm(all_spk_embedding, dim=1, keepdim=True) + eps)
        
        spk_vector = self.speaker_stack(mixture) # (batch_size, n_sources, latent_dim, T)
        spk_centroids = spk_vector.mean(dim=-1) # (batch_size, n_sources, latent_dim)

        estimated_sources = self.separation_stack(mixture, spk_centroids, return_all=return_all_layers, stack_dim=stack_dim)

        output = []
        output.append(estimated_sources)

        if return_spk_vector:    
            output.append(spk_vector)
        
        if return_all_spk_embedding:    
            output.append(all_spk_embedding)
        
        if len(output) == 1:
            output = output[0]
        else:
            output = tuple(output)
        
        return output

    def compute_pit_speaker_loss(self, spk_vector, spk_embedding, all_spk_embedding, feature_last=True, batch_mean=True):
        """
        Args:
            spk_vector: (batch_size, T, n_sources, latent_dim)
            spk_embedding: (batch_size, n_sources, latent_dim)
            all_spk_embedding: (batch_size, n_training_sources, latent_dim)
        Returns:
            loss: (batch_size,) or ()
            pattern: (batch_size, n_sources)
        """
        assert feature_last, "feature_last should be True."

        patterns = list(itertools.permutations(range(self.n_sources)))
        patterns = torch.Tensor(patterns).long()
        
        P = len(patterns)
        possible_loss = []
        
        for idx in range(P):
            pattern = patterns[idx]
            loss = self.spk_criterion(spk_vector[:, :, pattern], spk_embedding, all_spk_embedding, feature_last=feature_last, batch_mean=False, time_mean=True) # (batch_size,)
            possible_loss.append(loss)
        
        possible_loss = torch.stack(possible_loss, dim=1) # (batch_size, P)
        loss, indices = torch.min(possible_loss, dim=1) # loss (batch_size,), indices (batch_size,)
        
        if batch_mean:
            loss = loss.mean(dim=0)
        
        pattern = patterns[indices]
        
        return loss, pattern

class _SpeakerDistance(nn.Module):
    def __init__(self, n_sources):
        super().__init__()

        self.mask = nn.Parameter(1 - torch.eye(n_sources), requires_grad=False)

        zero_dim_size = ()
        self.scale, self.bias = nn.Parameter(torch.empty(zero_dim_size), requires_grad=True), nn.Parameter(torch.empty(zero_dim_size), requires_grad=True)

        self._reset_parameters()
    
    def _reset_parameters(self):
        self.scale.data.fill_(1)
        self.bias.data.fill_(0)

    def forward(self, spk_vector, spk_embedding, _, feature_last=True, batch_mean=True, time_mean=True):
        """
        Args:
            spk_vector:
                (batch_size, T, n_sources, latent_dim) if feature_last
                (batch_size, n_sources, latent_dim, T) otherwise
            spk_embedding: (batch_size, n_sources, latent_dim)
            _: All speaker embedding (n_training_sources, latent_dim)
        Returns:
            loss: (batch_size, T) or (T,)
        """
        if not feature_last:
            spk_vector = spk_vector.permute(0, 3, 1, 2).contiguous() # (batch_size, T, n_sources, latent_dims)

        loss_euclid = self.compute_euclid_distance(spk_vector, spk_embedding.unsqueeze(dim=1), dim=-1) # (batch_size, T, n_sources)

        distance_table = self.compute_euclid_distance(spk_vector.unsqueeze(dim=3), spk_vector.unsqueeze(dim=2), dim=-1) # (batch_size, T, n_sources, n_sources)
        loss_hinge = F.relu(1 - distance_table) # (batch_size, T, n_sources, n_sources)
        loss_hinge = torch.sum(self.mask * loss_hinge, dim=2) # (batch_size, T, n_sources)

        loss = loss_euclid + loss_hinge
        loss = loss.mean(dim=-1)

        if time_mean:
            loss = loss.mean(dim=1)
        
        if batch_mean:
            loss = loss.mean(dim=0)
        return loss
    
    def compute_euclid_distance(self, input, target, dim=-1, keepdim=False, scale=None, bias=None):
        distance = torch.sum((input - target)**2, dim=dim, keepdim=keepdim)

        if scale is not None or bias is not None:
            distance = torch.abs(self.scale) * distance + self.bias
        
        return distance

class _SpeakerLoss(nn.Module):
    def __init__(self, n_sources):
        super().__init__()

        self.mask = nn.Parameter(1 - torch.eye(n_sources), requires_grad=False)

        zero_dim_size = ()
        self.scale, self.bias = nn.Parameter(torch.empty(zero_dim_size), requires_grad=True), nn.Parameter(torch.empty(zero_dim_size), requires_grad=True)

        self._reset_parameters()
    
    def _reset_parameters(self):
        self.scale.data.fill_(1)
        self.bias.data.fill_(0)

    def forward(self, speaker_vector, speaker_embedding, all_speaker_embedding, feature_last=True, batch_mean=True, time_mean=True):
        """
        Args:
            speaker_vector:
                (batch_size, T, n_sources, latent_dims) if feature_last=True
                (batch_size, n_sources, latent_dims, T) otherwise
            speaker_embedding: (batch_size, n_sources, latent_dim)
            all_speaker_embedding: (n_training_sources, latent_dim)
        Returns:
            loss: (batch_size,) or ()
        """
        if not feature_last:
            speaker_vector = speaker_vector.permute(0, 3, 1, 2).contiguous() # (batch_size, T, n_sources, latent_dims)
        
        loss = self.compute_speaker_loss(speaker_vector, speaker_embedding, all_speaker_embedding, scale=self.scale, bias=self.bias, feature_last=True, batch_mean=False) # (batch_size, T, n_sources)
        loss = loss.mean(dim=-1)

        if time_mean:
            loss = loss.mean(dim=1)
        
        if batch_mean:
            loss = loss.mean(dim=0)

        return loss
    
    def compute_speaker_loss(self, spk_vector, spk_embedding, all_spk_embedding, scale=None, bias=None, feature_last=True, batch_mean=True):
        """
        Args:
            spk_vector: (batch_size, T, n_sources, latent_dim)
            spk_embedding: (batch_size, n_sources, latent_dim)
            all_spk_embedding: (n_training_sources, latent_dim)
        Returns:
            loss: (batch_size, T, n_sources) or (T, n_sources)
        """
        assert feature_last, "feature_last should be True."

        loss_distance = self.compute_speaker_distance(spk_vector, spk_embedding, feature_last=feature_last, batch_mean=False) # (batch_size, T, n_sources)

        rescaled_distance = self.compute_euclid_distance(spk_vector, spk_embedding.unsqueeze(dim=1), dim=-1, scale=scale, bias=bias) # (batch_size, T, n_sources)
        rescaled_all_distance = self.compute_euclid_distance(spk_vector.unsqueeze(dim=3), all_spk_embedding, dim=-1, scale=scale, bias=bias) # (batch_size, T, n_sources, n_training_sources)

        loss_local = self.compute_local_classification(rescaled_distance, batch_mean=False) # (batch_size, T, n_sources)
        loss_global = self.compute_global_classification(rescaled_distance, rescaled_all_distance, batch_mean=False) # (batch_size, T, n_sources)
        loss = loss_distance + loss_local + loss_global

        if batch_mean:
            loss = loss.mean(dim=0)

        return loss
    
    def compute_speaker_distance(self, spk_vector, spk_embedding, feature_last=True, batch_mean=True):
        """
        Args:
            spk_vector: (batch_size, T, n_sources, latent_dim)
            spk_embedding: (batch_size, n_sources, latent_dim)
        Returns:
            loss: (batch_size, T, n_sources)
        """
        assert feature_last, "feature_last should be True."

        loss_distance = self.compute_euclid_distance(spk_vector, spk_embedding.unsqueeze(dim=1), dim=-1) # (batch_size, T, n_sources)

        distance_table = self.compute_euclid_distance(spk_vector.unsqueeze(dim=3), spk_vector.unsqueeze(dim=2), dim=-1) # (batch_size, T, n_sources, n_sources)
        loss_hinge = F.relu(1 - distance_table) # (batch_size, T, n_sources, n_sources)
        loss_hinge = torch.sum(self.mask * loss_hinge, dim=2) # (batch_size, T, n_sources)

        loss = loss_distance + loss_hinge

        if batch_mean:
            loss = loss.mean(dim=0)

        return loss
    
    def compute_local_classification(self, distance, batch_mean=True):
        """
        Args:
            distance: (batch_size, T, n_sources)
        Returns:
            loss: (batch_size, T, n_sources)
        """
        loss = distance + torch.logsumexp(- distance, dim=2, keepdim=True) # (batch_size, T, n_sources)

        if batch_mean:
            loss = loss.mean(dim=0)

        return loss

    def compute_global_classification(self, distance, all_distance, batch_mean=True):
        """
        Args:
            distance: (batch_size, T, n_sources)
            all_distance: (batch_size, T, n_sources, n_training_sources)
        Returns:
            loss: (batch_size, T, n_sources)
        """
        loss = distance + torch.logsumexp(- all_distance, dim=3) # (batch_size, T, n_sources)

        if batch_mean:
            loss = loss.mean(dim=0)

        return loss

    def compute_euclid_distance(self, input, target, dim=-1, keepdim=False, scale=None, bias=None):
        distance = torch.sum((input - target)**2, dim=dim, keepdim=keepdim)

        if scale is not None or bias is not None:
            distance = torch.abs(self.scale) * distance + self.bias
        
        return distance

def _test_wavesplit():
    batch_size = 4
    n_training_sources, n_sources = 10, 2
    in_channels = 1
    latent_dim = 8
    T = 1024

    input = torch.randn(batch_size, in_channels, T)
    target = torch.randn(batch_size, n_sources, T)
    spk_idx = torch.randint(0, n_training_sources, (batch_size, n_sources))

    spk_criterion = _SpeakerLoss(n_sources=n_sources)

    speaker_stack = SpeakerStack(in_channels, latent_dim, num_layers=5, separable=False, n_sources=n_sources)
    separation_stack = SeparationStack(in_channels, latent_dim, num_layers=5, separable=False, n_sources=n_sources)
    model = WaveSplit(speaker_stack, separation_stack, latent_dim, n_sources=n_sources, n_training_sources=n_training_sources, spk_criterion=spk_criterion)

    with torch.no_grad():
        sorted_idx = model(input, spk_idx=spk_idx, return_all_layers=False, return_spk_vector=False)
    
    model.zero_grad()

    estimated_sources, sorted_spk_vector = model(input, spk_idx=spk_idx, sorted_idx=sorted_idx, return_all_layers=True, return_spk_vector=True, return_spk_embedding=False, return_all_spk_embedding=False)

    loss = - sisdr(estimated_sources, target.unsqueeze(dim=1))
    loss = loss.mean()

    print(model)
    print(input.size(), estimated_sources.size(), sorted_idx.size(), sorted_spk_vector.size())

def _test_wavesplit_spk_distance():
    batch_size = 4
    n_training_sources, n_sources = 10, 2
    in_channels = 1
    latent_dim = 8
    T = 1024
    
    input = torch.randn(batch_size, in_channels, T)
    target = torch.randn(batch_size, n_sources, T)
    spk_idx = torch.randint(0, n_training_sources, (batch_size, n_sources))

    spk_criterion = _SpeakerDistance(n_sources=n_sources)

    speaker_stack = SpeakerStack(in_channels, latent_dim, num_layers=5, separable=False, n_sources=n_sources)
    separation_stack = SeparationStack(in_channels, latent_dim, num_layers=5, separable=False, n_sources=n_sources)
    model = WaveSplit(speaker_stack, separation_stack, latent_dim, n_sources=n_sources, n_training_sources=n_training_sources, spk_criterion=spk_criterion)

    with torch.no_grad():
        sorted_idx = model(input, spk_idx=spk_idx, return_all_layers=False, return_spk_vector=False)
    
    model.zero_grad()

    estimated_sources, sorted_spk_vector = model(input, spk_idx=spk_idx, sorted_idx=sorted_idx, return_all_layers=True, return_spk_vector=True, return_spk_embedding=False, return_all_spk_embedding=False)

    loss = - sisdr(estimated_sources, target.unsqueeze(dim=1))
    loss = loss.mean()

    print(model)
    print(model.num_parameters)
    print(input.size(), estimated_sources.size(), sorted_idx.size(), sorted_spk_vector.size())
    print()

    model.eval()
    estimated_sources, sorted_spk_vector = model(input, return_all_layers=False, return_spk_vector=True)

    print(input.size(), estimated_sources.size(), sorted_idx.size(), sorted_spk_vector.size())
    print()

if __name__ == '__main__':
    import torch

    from models.wavesplit import SpeakerStack, SeparationStack
    from criterion.sdr import sisdr

    torch.manual_seed(111)

    print("="*10, "WaveSplit", "="*10)
    _test_wavesplit()

    print("="*10, "WaveSplitOracle", "="*10)
    _test_wavesplit_spk_distance()