import torch
import torch.nn as nn
import torch.nn.functional as F

class SpeakerLoss(nn.Module):
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
    
    def compute_speaker_loss(self, speaker_vector, speaker_embedding, all_speaker_embedding, scale=None, bias=None, feature_last=True, batch_mean=True):
        """
        Args:
            speaker_vector: (batch_size, T, n_sources, latent_dim)
            speaker_embedding: (batch_size, n_sources, latent_dim)
            all_speaker_embedding: (n_training_sources, latent_dim)
        Returns:
            loss: (batch_size, T, n_sources) or (T, n_sources)
        """
        assert feature_last, "feature_last should be True."

        loss_distance = self.compute_speaker_distance(speaker_vector, speaker_embedding, feature_last=feature_last, batch_mean=False) # (batch_size, T, n_sources)

        rescaled_distance = self.compute_euclid_distance(speaker_vector, speaker_embedding.unsqueeze(dim=1), dim=-1, scale=scale, bias=bias) # (batch_size, T, n_sources)
        
        rescaled_all_distance = []

        for spk_embedding in all_speaker_embedding:
            d = self.compute_euclid_distance(speaker_vector, spk_embedding, dim=-1, scale=scale, bias=bias) # (batch_size, T, n_sources)
            rescaled_all_distance.append(d)
        
        rescaled_all_distance = torch.stack(rescaled_all_distance, dim=3) # (batch_size, T, n_sources, n_training_sources)

        loss_local = self.compute_local_classification(rescaled_distance, batch_mean=False) # (batch_size, T, n_sources)
        loss_global = self.compute_global_classification(rescaled_distance, rescaled_all_distance, batch_mean=False) # (batch_size, T, n_sources)
        loss = loss_distance + loss_local + loss_global

        if batch_mean:
            loss = loss.mean(dim=0)

        return loss
    
    def compute_speaker_distance(self, speaker_vector, speaker_embedding, feature_last=True, batch_mean=True):
        """
        Args:
            speaker_vector: (batch_size, T, n_sources, latent_dim)
            speaker_embedding: (batch_size, n_sources, latent_dim)
        Returns:
            loss: (batch_size, T, n_sources)
        """
        assert feature_last, "feature_last should be True."

        loss_distance = self.compute_euclid_distance(speaker_vector, speaker_embedding.unsqueeze(dim=1), dim=-1) # (batch_size, T, n_sources)

        distance_table = self.compute_euclid_distance(speaker_vector.unsqueeze(dim=3), speaker_vector.unsqueeze(dim=2), dim=-1) # (batch_size, T, n_sources, n_sources)
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

class MultiDomainLoss(nn.Module):
    def __init__(self, criterion_reconst, criterion_speaker):
        super().__init__()

        self.criterion_reconst, self.criterion_speaker = criterion_reconst, criterion_speaker
    
    def forward(self, input, target, spk_vector=None, spk_embedding=None, all_spk_embedding=None, batch_mean=True):
        loss_reconst = self.criterion_reconst(input, target, batch_mean=batch_mean)
        loss_speaker = self.criterion_speaker(spk_vector, spk_embedding, all_spk_embedding, feature_last=False, batch_mean=batch_mean)
        loss = loss_reconst + loss_speaker
        
        return loss