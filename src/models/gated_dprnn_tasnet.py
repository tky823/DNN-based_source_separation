import torch
import torch.nn as nn
import torch.nn.functional as F

from models.tasnet import Encoder
from models.dprnn_tasnet import Segment1d, OverlapAdd1d
from models.mulcat_rnn import MulCatDPRNN

EPS=1e-12

"""
"Voice Separation with an Unknown Number of Multiple Speakers"
https://arxiv.org/abs/2003.01531

We have to name the model.
"""

class GatedDPRNNTasNet(nn.Module):
    def __init__(self, n_bases, kernel_size, stride=None, sep_hidden_channels=256, sep_chunk_size=100, sep_hop_size=50, sep_num_blocks=6, causal=False, n_sources=2, eps=EPS, **kwargs):
        super().__init__()
        
        if stride is None:
            stride = kernel_size//2
        
        assert kernel_size%stride == 0, "kernel_size is expected divisible by stride"
        
        # Encoder-decoder
        self.n_bases = n_bases
        self.kernel_size, self.stride = kernel_size, stride
               
        self.enc_nonlinear = kwargs['enc_nonlinear']
        
        # Separator configuration
        self.sep_hidden_channels = sep_hidden_channels
        self.sep_chunk_size, self.sep_hop_size = sep_chunk_size, sep_hop_size
        self.sep_num_blocks = sep_num_blocks
        
        self.n_sources = n_sources
        self.eps = eps
        
        # Network configuration
        self.encoder = Encoder(1, n_bases, kernel_size=kernel_size, stride=stride)
        self.separator = Separator(n_bases, hidden_channels=sep_hidden_channels, chunk_size=sep_chunk_size, hop_size=sep_hop_size, num_blocks=sep_num_blocks, causal=causal, n_sources=n_sources, eps=eps)
        self.decoder = OverlapAddDecoder(n_bases, 1, kernel_size=kernel_size, stride=stride)
        
        self.num_parameters = self._get_num_parameters()
        
    def forward(self, input):
        output, latent = self.extract_latent(input)
        
        return output
        
    def extract_latent(self, input):
        """
        Args:
            input (batch_size, 1, T)
        Returns:
            output (batch_size, n_sources, T)
            latent (batch_size, n_sources, n_bases, T'), where T' = (T-K)//S+1
        """
        n_sources = self.n_sources
        n_bases = self.n_bases
        kernel_size, stride = self.kernel_size, self.stride
        num_blocks_half = self.sep_num_blocks//2
        
        batch_size, C_in, T = input.size()
        
        assert C_in == 1, "input.size() is expected (?,1,?), but given {}".format(input.size())
        
        padding = (stride - (T-kernel_size)%stride)%stride
        padding_left = padding//2
        padding_right = padding - padding_left

        input = F.pad(input, (padding_left, padding_right))
        x = self.encoder(input)
        latent = self.separator(x) # (batch_size, num_blocks//2, n_sources, num_features, n_frames)    
        x = latent.view(batch_size, num_blocks_half*n_sources, n_bases, -1) / kernel_size
        x_hat = self.decoder(x)
        x_hat = x_hat.view(batch_size, num_blocks_half, n_sources, -1)
        output = F.pad(x_hat, (-padding_left, -padding_right))
        
        return output, latent
    
    def _get_num_parameters(self):
        num_parameters = 0
        
        for p in self.parameters():
            if p.requires_grad:
                num_parameters += p.numel()
                
        return num_parameters


class Separator(nn.Module):
    def __init__(self, num_features, hidden_channels=128, chunk_size=100, hop_size=50, num_blocks=6, causal=True, n_sources=2, eps=EPS):
        super().__init__()
        
        self.num_features, self.n_sources = num_features, n_sources
        self.chunk_size, self.hop_size = chunk_size, hop_size
        self.num_blocks = num_blocks
        
        self.segment1d = Segment1d(chunk_size, hop_size)
        self.mulcat_dprnn = MulCatDPRNN(num_features, hidden_channels, num_blocks=num_blocks, causal=causal, n_sources=n_sources, eps=eps)
        self.overlap_add1d = OverlapAdd1d(chunk_size, hop_size)

    def forward(self, input):
        """
        Args:
            input (batch_size, num_features, n_frames)
        Returns:
            output (batch_size, n_sources, num_features, n_frames)
        """
        num_features, n_sources = self.num_features, self.n_sources
        chunk_size, hop_size = self.chunk_size, self.hop_size
        num_blocks_half = self.num_blocks//2
        batch_size, num_features, n_frames = input.size()
        
        padding = (hop_size-(n_frames-chunk_size)%hop_size)%hop_size
        padding_left = padding//2
        padding_right = padding - padding_left
        
        x = F.pad(input, (padding_left, padding_right))
        x = self.segment1d(x)
        x = self.mulcat_dprnn(x) # (batch_size, num_blocks//2, n_sources, num_features, S, chunk_size)
        x = x.view(batch_size*num_blocks_half*n_sources, num_features, -1, chunk_size)
        x = self.overlap_add1d(x)
        x = F.pad(x, (-padding_left, -padding_right))
        output = x.view(batch_size, num_blocks_half, n_sources, num_features, n_frames)
        
        return output

class OverlapAddDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=None):
        super().__init__()

    def forward(self, input):
        return input


if __name__ == '__main__':
    batch_size = 4

    num_features, H = 8, 32
    n_frames = 20
    K, P = 4, 2
    B = 6
    causal = False
    n_sources = 2

    input = torch.randint(0, 10, (batch_size, num_features, n_frames), dtype=torch.float)

    model = Separator(num_features, H, chunk_size=K, hop_size=P, num_blocks=B, causal=causal, n_sources=n_sources)
    print(model)

    output = model(input)
    print(input.size(), output.size())
    print()

    N, L = 128, 8
    H = 256
    K, P = 100, 50
    B = 6
    C, T = 1, 1024

    input = torch.randint(0, 10, (batch_size, C, T), dtype=torch.float)
    
    model = GatedDPRNNTasNet(N, L, sep_hidden_channels=H, sep_chunk_size=K, sep_hop_size=P, sep_num_blocks=B, n_sources=2, enc_nonlinear=None)

    print(model)
    print(model)
    print("# Parameters: {}".format(model.num_parameters))

    output = model(input)
    print(input.size(), output.size())
    print()