import torch.nn as nn
import torch.nn.functional as F

from utils.utils_tasnet import choose_layer_norm
from models.film import FiLM1d

EPS = 1e-12

class WaveSplit(nn.Module):
    def __init__(self, in_channels, latent_dim=512, kernel_size=3, sep_num_blocks=4, sep_num_layers=10, spk_num_layers=14, dilated=True, separable=True, causal=False, nonlinear=None, norm=True, n_sources=2, n_training_sources=None, eps=EPS):
        super().__init__()

        self.embed_sources = nn.Embedding(n_training_sources, latent_dim)
        self.mask = nn.Parameter(1 - torch.eye(n_sources), requeires_grad=False)

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
    
    def forward(self, input, speaker_id=None):
        """
        Args:
            input (batch_size, 1, T)
        Returns:
            output (batch_size, 1, T)
        """
        speaker_vector = self.speaker_stack(input) # (batch_size, n_sources, latent_dim, T) TODO
        speaker_centroids = speaker_vector.mean(dim=-1) # (batch_size, n_sources, latent_dim) for experimentally
        output = self.sepatation_stack(input, speaker_centroids)

        return output
    
    def compute_speaker_loss(self, speaker_vector, speaker_id, batch_mean=True):
        """
        Args:
            speaker_vector: (batch_size, n_sources, latent_dim, T)
            speaker_id: (batch_size, n_sources)
        Returns:
            loss: (batch_size,) or ()
        """
        loss_distance = self.compute_speaker_distance(speaker_vector, speaker_id)
        loss_local = 0 # self.compute_local_classification(speaker_vector, speaker_id)
        loss_global = 0 # self.compute_global_classification(speaker_vector, speaker_id)

        return loss_distance + loss_local + loss_global
    
    def compute_speaker_distance(self, speaker_vector, speaker_id):
        """
        Args:
            speaker_vector: (batch_size, n_sources, latent_dim, T)
            speaker_id: (batch_size, n_sources)
        Returns:
            loss: (batch_size, T) or (T,)
        """
        mask = self.mask # (n_sources, n_sources)
        embedding = self.embed_sources(speaker_id) # (batch_size, n_sources, latent_dim)

        loss = (speaker_vector - embedding.unsqueeze(dim=-1))**2
        loss = loss.sum(dim=2) # (batch_size, n_sources, T)

        loss_table = speaker_vector.unsqueeze(dim=2) - speaker_vector.unsqueeze(dim=1)
        loss_table = F.relu(1 - torch.sum(loss_table**2, dim=-2)) # (batch_size, n_sources, n_sources, T)
        loss_table = torch.sum(mask.unsqueeze(dim=2) * loss_table, dim=2) # (batch_size, n_sources, T)

        return loss + loss_table

    def compute_local_classification(self, speaker_vector, speaker_id):
        return 0

    def compute_global_classification(self, speaker_vector, speaker_id):
        return 0


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
                    _out_channels = latent_dim
                elif block_idx == num_blocks - 1 and layer_idx == num_layers - 1:
                    residual = False
                    _in_channels = latent_dim
                    _out_channels = n_sources
                else:
                    residual = True
                    _in_channels = latent_dim
                    _out_channels = latent_dim
                
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

    def forward(self, input, speaker_centroids):
        """
        Args:
            input (batch_size, in_channels, T)
            speaker_centroids (batch_size, n_sources, latent_dim)
        """
        n_sources = self.n_sources
        x = input
        
        for block_idx in range(self.num_blocks):
            fc_weights_block = self.fc_weights[block_idx]
            fc_biases_block = self.fc_biases[block_idx]
            net_block = self.net[block_idx]
            for layer_idx in range(self.num_layers):
                gamma = fc_weights_block[layer_idx](speaker_centroids)
                beta = fc_biases_block[layer_idx](speaker_centroids)
                x = net_block[layer_idx](x, gamma, beta)
        
        output = x

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
    n_sources = 2
    in_channels = 1
    latent_dim = 512
    T = 1024
    input = torch.randn(batch_size, in_channels, T)

    model = WaveSplit(in_channels, latent_dim, n_sources=n_sources)
    output = model(input)

    print(input.size(), output.size())

if __name__ == '__main__':
    import torch

    torch.manual_seed(111)

    print("="*10, "Modules for WaveSplit", "="*10)
    _test()

    print("="*10, "WaveSplit", "="*10)
    _test_wavesplit()