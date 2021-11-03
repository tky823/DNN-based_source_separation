import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils_tasnet import choose_layer_norm
from models.film import FiLM1d

EPS = 1e-12

class WaveSplitBase(nn.Module):
    def __init__(
        self, in_channels, latent_dim=512,
        kernel_size=3, sep_num_blocks=4, sep_num_layers=10, spk_num_layers=14,
        dilated=True, separable=True, causal=False, nonlinear=None, norm=True,
        n_sources=2, n_training_sources=None,
        eps=EPS
    ):
        super().__init__()

        self.embed_sources = nn.Embedding(n_training_sources, latent_dim)

        self.speaker_stack = SpeakerStack(
            in_channels, latent_dim=latent_dim,
            kernel_size=kernel_size, num_layers=spk_num_layers,
            dilated=dilated, separable=separable, causal=causal, nonlinear=nonlinear, norm=norm,
            n_sources=n_sources,
            eps=eps
        )

        self.sepatation_stack = SeparationStack(
            in_channels, latent_dim=latent_dim,
            kernel_size=kernel_size, num_blocks=sep_num_blocks, num_layers=sep_num_layers,
            dilated=dilated, separable=separable, causal=causal, nonlinear=nonlinear, norm=norm,
            n_sources=n_sources,
            eps=eps
        )

        all_speaker_id = torch.arange(n_training_sources).long()
        self.all_speaker_id = nn.Parameter(all_speaker_id, requires_grad=False)

        self.n_sources, self.n_training_sources = n_sources, n_training_sources
    
    def forward(self, input, speaker_id=None):
        """
        Args:
            input (batch_size, 1, T)
        Returns:
            output (batch_size, 1, T)
        """
        raise NotImplementedError("Implement forward.")
    
    @property
    def num_parameters(self):
        _num_parameters = 0
        
        for p in self.parameters():
            if p.requires_grad:
                _num_parameters += p.numel()
                
        return _num_parameters

class WaveSplit(WaveSplitBase):
    def __init__(
        self,
        in_channels, latent_dim=512,
        kernel_size=3, sep_num_blocks=4, sep_num_layers=10, spk_num_layers=14,
        dilated=True, separable=True, causal=False, nonlinear=None, norm=True,
        n_sources=2, n_training_sources=None,
        spk_criterion=None,
        eps=EPS):
        super().__init__(
            in_channels, latent_dim=latent_dim,
            kernel_size=kernel_size,
            sep_num_blocks=sep_num_blocks, sep_num_layers=sep_num_layers, spk_num_layers=spk_num_layers,
            dilated=dilated, separable=separable, causal=causal,
            nonlinear=nonlinear, norm=norm,
            n_sources=n_sources, n_training_sources=n_training_sources,
            eps=eps
        )

        self.spk_criterion = spk_criterion
    
    def forward(self, input, speaker_id=None, return_all=False, return_speaker_vector=False, stack_dim=1):
        """
        Only supports training time
        Args:
            input (batch_size, 1, T)
        Returns:
            output (batch_size, num_blocks * num_layers, 1, T) if stack_dim=1.
            sorted_speaker_vector: (batch_size, n_sources, latent_dim, T)
        """
        output, sorted_speaker_vector = self.extract_latent(input, speaker_id=speaker_id, return_all=return_all, stack_dim=stack_dim)

        if return_speaker_vector:
            return output, sorted_speaker_vector

        return output
    
    def extract_latent(self, input, speaker_id=None, return_all=False, stack_dim=1):
        """
        Only supports training time
        Args:
            input: (batch_size, 1, T)
        Returns:
            output: (batch_size, num_blocks * num_layers, 1, T) if stack_dim=1.
            sorted_speaker_vector: (batch_size, n_sources, latent_dim, T)
        """
        speaker_vector = self.speaker_stack(input) # (batch_size, n_sources, latent_dim, T)

        batch_size, n_sources, latent_dim, T = speaker_vector.size()
        speaker_vector = speaker_vector.permute(0, 3, 1, 2).contiguous() # (batch_size, T, n_sources, latent_dim)

        if self.training:
            with torch.no_grad():
                speaker_embedding = self.embed_sources(speaker_id) # (batch_size, n_sources, latent_dim)
                all_speaker_embedding = self.embed_sources(self.all_speaker_id) # (n_training_sources, latent_dim)

                _, sorted_idx = self.compute_pit_speaker_loss(speaker_vector, speaker_embedding, all_speaker_embedding, feature_last=True, batch_mean=False) # (batch_size, T, n_sources)

                sorted_idx = sorted_idx.view(batch_size * T, n_sources)
                flatten_sorted_idx = sorted_idx + torch.arange(0, batch_size * T * n_sources, n_sources).long().unsqueeze(dim=-1)
                flatten_sorted_idx = flatten_sorted_idx.view(batch_size * T * n_sources)

            flatten_speaker_vector = speaker_vector.view(batch_size * T * n_sources, latent_dim)
            flatten_speaker_vector = flatten_speaker_vector[flatten_sorted_idx]
            sorted_speaker_vector = flatten_speaker_vector.view(batch_size, T, n_sources, latent_dim)
            sorted_speaker_vector = sorted_speaker_vector.permute(0, 2, 3, 1).contiguous() # (batch_size, n_sources, latent_dim, T)
            speaker_centroids = sorted_speaker_vector.mean(dim=-1) # (batch_size, n_sources, latent_dim)
        else:
            raise NotImplementedError("Not support test time process.")
        
        output = self.sepatation_stack(input, speaker_centroids, return_all=return_all, stack_dim=stack_dim)

        return output, sorted_speaker_vector
    
    def compute_pit_speaker_loss(self, speaker_vector, speaker_embedding, all_speaker_embedding, feature_last=True, batch_mean=True):
        """
        Args:
            speaker_vector: (batch_size, T, n_sources, latent_dim)
            speaker_embedding: (batch_size, n_sources, latent_dim)
            all_speaker_embedding: (batch_size, n_training_sources, latent_dim)
        Returns:
            loss: (batch_size, T) or (T,)
        """
        assert feature_last, "feature_last should be True."

        patterns = list(itertools.permutations(range(self.n_sources)))
        patterns = torch.Tensor(patterns).long()
        
        P = len(patterns)
        possible_loss = []
        
        for idx in range(P):
            pattern = patterns[idx]
            loss = self.spk_criterion(speaker_vector[:, :, pattern], speaker_embedding, all_speaker_embedding, feature_last=feature_last, batch_mean=False, time_mean=False) # (batch_size, T)
            possible_loss.append(loss)
        
        possible_loss = torch.stack(possible_loss, dim=2) # (batch_size, T, P)
        loss, indices = torch.min(possible_loss, dim=2) # loss (batch_size, T), indices (batch_size, T)
        
        if batch_mean:
            loss = loss.mean(dim=0)
        
        return loss, patterns[indices]

    @property
    def num_parameters(self):
        _num_parameters = 0

        for p in self.embed_sources.parameters():
            if p.requires_grad:
                _num_parameters += p.numel()
        
        for p in self.speaker_stack.parameters():
            if p.requires_grad:
                _num_parameters += p.numel()

        for p in self.sepatation_stack.parameters():
            if p.requires_grad:
                _num_parameters += p.numel()
                
        return _num_parameters

class SpeakerStack(nn.Module):
    def __init__(self, in_channels, latent_dim=512, kernel_size=3, num_layers=14, dilated=True, separable=True, causal=False, nonlinear=None, norm=True, n_sources=2, eps=EPS):
        super().__init__()
        
        self.num_layers = num_layers
        self.n_sources = n_sources
        
        net = []
        
        for idx in range(num_layers):
            if dilated:
                dilation = 2**idx
                stride = 1
            else:
                dilation = 1
                stride = 2
            
            if idx == 0:
                residual = True
                _in_channels = in_channels
                _out_channels = latent_dim
            elif idx == num_layers - 1:
                residual = False
                _in_channels = latent_dim
                _out_channels = latent_dim * n_sources
            else:
                residual = True
                _in_channels = latent_dim
                _out_channels = latent_dim
            
            block = ConvBlock1d(
                _in_channels, _out_channels, hidden_channels=latent_dim,
                kernel_size=kernel_size, stride=stride, dilation=dilation,
                separable=separable, causal=causal, nonlinear=nonlinear, norm=norm,
                residual=residual,
                eps=eps
            )
            net.append(block)
        
        self.net = nn.Sequential(*net)

    def forward(self, input):
        n_sources = self.n_sources
        x = input
        
        for idx in range(self.num_layers):
            x = self.net[idx](x)
        
        batch_size, _, T = x.size()
        output = x.view(batch_size, n_sources, -1, T)

        return output

class SeparationStack(nn.Module):
    def __init__(self, in_channels, latent_dim=512, kernel_size=3, num_blocks=4, num_layers=10, dilated=True, separable=True, causal=False, nonlinear=None, norm=True, n_sources=2, eps=EPS):
        super().__init__()
        
        self.num_blocks, self.num_layers = num_blocks, num_layers
        self.n_sources = n_sources
        
        net = []
        fc_weights, fc_biases = [], []

        for block_idx in range(num_blocks):
            subnet = []
            sub_fc_weights, sub_fc_biases = [], []

            for layer_idx in range(num_layers):
                if dilated:
                    dilation = 2**layer_idx
                    stride = 1
                else:
                    dilation = 1
                    stride = 2
                
                if block_idx == 0 and layer_idx == 0:
                    residual = True
                    _in_channels = in_channels
                elif block_idx == num_blocks - 1 and layer_idx == num_layers - 1:
                    residual = False
                    _in_channels = n_sources
                else:
                    residual = True
                    _in_channels = n_sources
                _out_channels = n_sources
                
                block = FiLMConvBlock1d(
                    _in_channels, _out_channels, hidden_channels=latent_dim,
                    kernel_size=kernel_size, stride=stride, dilation=dilation,
                    separable=separable, causal=causal, nonlinear=nonlinear, norm=norm,
                    residual=residual,
                    eps=eps
                )
                subnet.append(block)

                # For FiLM
                weights = MultiSourceProjection1d(latent_dim, _out_channels, n_sources=n_sources)
                biases = MultiSourceProjection1d(latent_dim, _out_channels, n_sources=n_sources)
                sub_fc_weights.append(weights)
                sub_fc_biases.append(biases)
            
            subnet = nn.Sequential(*subnet)
            net.append(subnet)

            sub_fc_weights, sub_fc_biases = nn.ModuleList(sub_fc_weights), nn.ModuleList(sub_fc_biases)
            fc_weights.append(sub_fc_weights)
            fc_biases.append(sub_fc_biases)
        
        self.fc_weights = nn.ModuleList(fc_weights)
        self.fc_biases = nn.ModuleList(fc_biases)

        self.net = nn.Sequential(*net)

    def forward(self, input, speaker_centroids, return_all=False, stack_dim=1):
        """
        Args:
            input (batch_size, in_channels, T)
            speaker_centroids (batch_size, n_sources, latent_dim)
        """
        x = input
        output = []
        
        for block_idx in range(self.num_blocks):
            fc_weights_block = self.fc_weights[block_idx]
            fc_biases_block = self.fc_biases[block_idx]
            net_block = self.net[block_idx]
            for layer_idx in range(self.num_layers):
                gamma = fc_weights_block[layer_idx](speaker_centroids)
                beta = fc_biases_block[layer_idx](speaker_centroids)
                x = net_block[layer_idx](x, gamma, beta)
                output.append(x)
        
        if return_all:
            output = torch.stack(output, dim=stack_dim)
        else:
            output = output[-1]

        return output

class ConvBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=256, kernel_size=3, stride=1, dilation=1, separable=True, causal=False, nonlinear=None, norm=True, residual=True, eps=EPS):
        super().__init__()
        
        self.kernel_size, self.stride, self.dilation = kernel_size, stride, dilation
        self.separable, self.causal = separable, causal
        self.norm = norm
        self.residual = residual
        
        self.bottleneck_conv1d = nn.Conv1d(in_channels, hidden_channels, kernel_size=1, stride=1)
        
        if nonlinear is not None:
            if nonlinear == 'prelu':
                self.nonlinear1d = nn.PReLU()
            else:
                raise ValueError("Not support {}".format(nonlinear))
            self.nonlinear = True
        else:
            self.nonlinear = False
        
        if norm:
            norm_name = 'cLN' if causal else 'gLN'
            self.norm1d = choose_layer_norm(norm_name, hidden_channels, causal=causal, eps=eps)
        if separable:
            self.separable_conv1d = DepthwiseSeparableConv1d(hidden_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, causal=causal, nonlinear=nonlinear, norm=norm, eps=eps)
        else:
            self.conv1d = nn.Conv1d(hidden_channels, out_channels, kernel_size=kernel_size, dilation=dilation)
            
    def forward(self, input):
        kernel_size, stride, dilation = self.kernel_size, self.stride, self.dilation
        nonlinear, norm = self.nonlinear, self.norm
        separable, causal = self.separable, self.causal
        
        _, _, T_original = input.size()
        
        residual = input
        x = self.bottleneck_conv1d(input)
        
        if nonlinear:
            x = self.nonlinear1d(x)
        if norm:
            x = self.norm1d(x)
        
        padding = (T_original - 1) * stride - T_original + (kernel_size - 1) * dilation + 1
        
        if causal:
            padding_left = padding
            padding_right = 0
        else:
            padding_left = padding // 2
            padding_right = padding - padding_left

        x = F.pad(x, (padding_left, padding_right))
        
        if separable:
            output = self.separable_conv1d(x)
        else:
            output = self.conv1d(x)
        
        if self.residual:
            output = output + residual
            
        return output

class FiLMConvBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=256, kernel_size=3, stride=1, dilation=1, separable=True, causal=False, nonlinear=None, norm=True, residual=True, eps=EPS):
        super().__init__()
        
        self.kernel_size, self.stride, self.dilation = kernel_size, stride, dilation
        self.separable, self.causal = separable, causal
        self.norm = norm
        self.residual = residual
        
        self.bottleneck_conv1d = nn.Conv1d(in_channels, hidden_channels, kernel_size=1, stride=1)
        
        if nonlinear is not None:
            if nonlinear == 'prelu':
                self.nonlinear1d = nn.PReLU()
            else:
                raise ValueError("Not support {}".format(nonlinear))
            self.nonlinear = True
        else:
            self.nonlinear = False
        
        if norm:
            norm_name = 'cLN' if causal else 'gLN'
            self.norm1d = choose_layer_norm(norm_name, hidden_channels, causal=causal, eps=eps)
        if separable:
            self.separable_conv1d = DepthwiseSeparableConv1d(hidden_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, causal=causal, nonlinear=nonlinear, norm=norm, eps=eps)
        else:
            self.conv1d = nn.Conv1d(hidden_channels, out_channels, kernel_size=kernel_size, dilation=dilation)
        
        self.film1d = FiLM1d()
            
    def forward(self, input, gamma, beta):
        kernel_size, stride, dilation = self.kernel_size, self.stride, self.dilation
        nonlinear, norm = self.nonlinear, self.norm
        separable, causal = self.separable, self.causal
        
        _, _, T_original = input.size()
        
        residual = input
        x = self.bottleneck_conv1d(input)
        
        if nonlinear:
            x = self.nonlinear1d(x)
        if norm:
            x = self.norm1d(x)
        
        padding = (T_original - 1) * stride - T_original + (kernel_size - 1) * dilation + 1
        
        if causal:
            padding_left = padding
            padding_right = 0
        else:
            padding_left = padding // 2
            padding_right = padding - padding_left

        x = F.pad(x, (padding_left, padding_right))
        
        if separable:
            x = self.separable_conv1d(x)
        else:
            x = self.conv1d(x)
        
        output = self.film1d(x, gamma, beta)
        
        if self.residual:
            output = output + residual
            
        return output

class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels=256, kernel_size=3, stride=1, dilation=1, causal=False, nonlinear=None, norm=True, eps=EPS):
        super().__init__()
        
        self.norm = norm
        self.eps = eps
        
        self.depthwise_conv1d = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, groups=in_channels)
        
        if nonlinear is not None:
            if nonlinear == 'prelu':
                self.nonlinear1d = nn.PReLU()
            else:
                raise ValueError("Not support {}".format(nonlinear))
            self.nonlinear = True
        else:
            self.nonlinear = False
        
        if norm:
            norm_name = 'cLN' if causal else 'gLN'
            self.norm1d = choose_layer_norm(norm_name, in_channels, causal=causal, eps=eps)
        
        self.pointwise_conv1d = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1)
        
    def forward(self, input):
        nonlinear, norm = self.nonlinear, self.norm
        
        x = self.depthwise_conv1d(input)
        
        if nonlinear:
            x = self.nonlinear1d(x)
        if norm:
            x = self.norm1d(x)
        
        output = self.pointwise_conv1d(x)
        
        return output

class MultiSourceProjection1d(nn.Module):
    def __init__(self, in_channels, out_channels, n_sources, channel_last=True):
        super().__init__()
        
        assert channel_last, "channel_last should be True."
        self.linear = nn.Linear(n_sources * in_channels, out_channels)
    
    def forward(self, input):
        batch_size, n_sources, in_channels = input.size()

        x = input.view(batch_size, n_sources*in_channels)
        output = self.linear(x)

        return output

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
            speaker_vector: (batch_size, n_sources, latent_dim, T)
            speaker_embedding: (batch_size, n_sources, latent_dim)
            all_speaker_embedding: (n_training_sources, latent_dim)
        Returns:
            loss: (batch_size,) or ()
        """
        loss = self.compute_speaker_loss(speaker_vector, speaker_embedding, all_speaker_embedding, scale=self.scale, bias=self.bias, feature_last=feature_last, batch_mean=False) # (batch_size, T, n_sources)
        
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
        rescaled_all_distance = self.compute_euclid_distance(speaker_vector.unsqueeze(dim=3), all_speaker_embedding, dim=-1, scale=scale, bias=bias) # (batch_size, T, n_sources, n_training_sources)

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

def _test():
    in_channels = 3
    input = torch.randn(4, in_channels, 128)

    model = DepthwiseSeparableConv1d(in_channels)
    output = model(input)

    print(input.size(), output.size())

    in_channels, out_channels = 3, 32
    input = torch.randn(4, in_channels, 128)

    model = ConvBlock1d(in_channels, out_channels, residual=False)
    output = model(input)

    print(input.size(), output.size())

    in_channels = 1
    input = torch.randn(4, in_channels, 128)

    model = SpeakerStack(in_channels)
    output = model(input)

    print(input.size(), output.size())

    batch_size = 4
    n_sources = 2
    in_channels = 1
    latent_dim = 512
    T = 1024
    input = torch.randn(batch_size, in_channels, T)
    speaker_centroids = torch.randn(batch_size, n_sources, latent_dim)

    model = SeparationStack(in_channels, latent_dim, n_sources=n_sources)
    output = model(input, speaker_centroids)

    print(input.size(), output.size())

def _test_wavesplit():
    batch_size = 4
    n_training_sources, n_sources = 10, 2
    in_channels = 1
    latent_dim = 512
    T = 1024
    input = torch.randn(batch_size, in_channels, T)
    target = torch.randn(batch_size, n_sources, T)
    speaker_id = torch.randint(0, n_training_sources, (batch_size, n_sources))

    spk_criterion = SpeakerLoss(n_sources=n_sources)
    model = WaveSplit(in_channels, latent_dim, n_sources=n_sources, n_training_sources=n_training_sources, spk_criterion=spk_criterion)
    output, sorted_speaker_vector = model(input, speaker_id=speaker_id, return_all=True, return_speaker_vector=True, stack_dim=1)

    loss = - sisdr(output, target.unsqueeze(dim=1))
    loss = loss.mean()

    print(input.size(), output.size(), sorted_speaker_vector.size())

if __name__ == '__main__':
    import torch

    from criterion.sdr import sisdr

    torch.manual_seed(111)

    print("="*10, "Modules for WaveSplit", "="*10)
    _test()
    print()

    print("="*10, "WaveSplit", "="*10)
    _test_wavesplit()