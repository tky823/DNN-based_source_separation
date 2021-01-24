import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils_tasnet import choose_bases, choose_layer_norm
from models.gtu import GTU1d
from models.dprnn_tasnet import Segment1d, OverlapAdd1d

EPS=1e-12

class DPTNet(nn.Module):
    def __init__(self, n_bases, kernel_size, stride=None, enc_bases=None, dec_bases=None, sep_hidden_channels=256, sep_chunk_size=100, sep_hop_size=50, sep_num_blocks=6, dilated=True, separable=True, causal=True, sep_norm=True, eps=EPS, mask_nonlinear='sigmoid', n_sources=2, **kwargs):
        super().__init__()
        
        if stride is None:
            stride = kernel_size//2
        
        assert kernel_size%stride == 0, "kernel_size is expected divisible by stride"
        
        # Encoder-decoder
        self.n_bases = n_bases
        self.kernel_size, self.stride = kernel_size, stride
        self.enc_bases, self.dec_bases = enc_bases, dec_bases
        
        if enc_bases == 'trainable' and not dec_bases == 'pinv':    
            self.enc_nonlinear = kwargs['enc_nonlinear']
        else:
            self.enc_nonlinear = None
        
        if enc_bases in ['Fourier', 'trainableFourier'] or dec_bases in ['Fourier', 'trainableFourier']:
            self.window_fn = kwargs['window_fn']
        else:
            self.window_fn = None
        
        # Separator configuration
        self.sep_hidden_channels = sep_hidden_channels
        self.sep_chunk_size, self.sep_hop_size = sep_chunk_size, sep_hop_size
        self.sep_num_blocks = sep_num_blocks
        
        self.dilated, self.separable, self.causal = dilated, separable, causal
        self.sep_norm = sep_norm
        self.mask_nonlinear = mask_nonlinear
        
        self.n_sources = n_sources
        self.eps = eps
        
        # Network configuration
        encoder, decoder = choose_bases(n_bases, kernel_size=kernel_size, stride=stride, enc_bases=enc_bases, dec_bases=dec_bases, **kwargs)
        
        self.encoder = encoder
        self.separator = Separator(n_bases, hidden_channels=sep_hidden_channels, chunk_size=sep_chunk_size, hop_size=sep_hop_size, num_blocks=sep_num_blocks, causal=causal, norm=sep_norm, mask_nonlinear=mask_nonlinear, n_sources=n_sources, eps=eps)
        self.decoder = decoder
        
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
        
        batch_size, C_in, T = input.size()
        
        assert C_in == 1, "input.size() is expected (?,1,?), but given {}".format(input.size())
        
        padding = (stride - (T-kernel_size)%stride)%stride
        padding_left = padding//2
        padding_right = padding - padding_left

        input = F.pad(input, (padding_left, padding_right))
        w = self.encoder(input)
        mask = self.separator(w)
        w = w.unsqueeze(dim=1)
        w_hat = w * mask
        latent = w_hat
        w_hat = w_hat.view(batch_size*n_sources, n_bases, -1)
        x_hat = self.decoder(w_hat)
        x_hat = x_hat.view(batch_size, n_sources, -1)
        output = F.pad(x_hat, (-padding_left, -padding_right))
        
        return output, latent
    
    def _get_num_parameters(self):
        num_parameters = 0
        
        for p in self.parameters():
            if p.requires_grad:
                num_parameters += p.numel()
                
        return num_parameters

class Separator(nn.Module):
    def __init__(self, num_features, hidden_channels=128, chunk_size=100, hop_size=50, num_blocks=6, causal=True, norm=True, mask_nonlinear='sigmoid', n_sources=2, eps=EPS):
        super().__init__()
        
        self.num_features, self.n_sources = num_features, n_sources
        self.chunk_size, self.hop_size = chunk_size, hop_size
        
        self.segment1d = Segment1d(chunk_size, hop_size)
        self.dprnn = DualPathTransformer(num_features, hidden_channels, num_blocks=num_blocks, causal=causal)
        self.overlap_add1d = OverlapAdd1d(chunk_size, hop_size)
        self.gtu = GTU1d(num_features, n_sources*num_features)
        
        if mask_nonlinear == 'relu':
            self.mask_nonlinear = nn.ReLU()
        elif mask_nonlinear == 'sigmoid':
            self.mask_nonlinear = nn.Sigmoid()
        elif mask_nonlinear == 'softmax':
            self.mask_nonlinear = nn.Softmax(dim=1)
        else:
            raise ValueError("Cannot support {}".format(mask_nonlinear))
            
    def forward(self, input):
        """
        Args:
            input (batch_size, num_features, T_bin)
        Returns:
            output (batch_size, n_sources, num_features, T_bin)
        """
        num_features, n_sources = self.num_features, self.n_sources
        chunk_size, hop_size = self.chunk_size, self.hop_size
        batch_size, num_features, T_bin = input.size()
        
        padding = (hop_size-(T_bin-chunk_size)%hop_size)%hop_size
        padding_left = padding//2
        padding_right = padding - padding_left
        
        x = F.pad(input, (padding_left, padding_right))
        x = self.segment1d(x)
        x = self.dprnn(x)
        x = self.overlap_add1d(x)
        x = F.pad(x, (-padding_left, -padding_right))
        # x = self.prelu(x)
        x = self.gtu(x)
        x = self.mask_nonlinear(x)
        output = x.view(batch_size, n_sources, num_features, T_bin)
        
        return output


class DualPathTransformer(nn.Module):
    def __init__(self, num_features, hidden_channels, num_blocks=6, causal=False, eps=EPS):
        super().__init__()
        
        # Network confguration
        net = []
        
        for _ in range(num_blocks):
            net.append(DualPathTransformerBlock(num_features, hidden_channels, causal=causal, eps=eps))
            
        self.net = nn.Sequential(*net)

    def forward(self, input):
        """
        Args:
            input (batch_size, num_features, S, chunk_size)
        Returns:
            output (batch_size, num_features, S, chunk_size)
        """
        output = self.net(input)

        return output

class DualPathTransformerBlock(nn.Module):
    def __init__(self, num_features, hidden_channels, causal, eps=EPS):
        super().__init__()
        
        self.intra_chunk_block = IntraChunkTransformer(num_features, hidden_channels, eps=eps)
        self.inter_chunk_block = InterChunkTransformer(num_features, hidden_channels, causal=causal, eps=eps)
        
    def forward(self, input):
        """
        Args:
            input (batch_size, num_features, S, chunk_size)
        Returns:
            output (batch_size, num_features, S, chunk_size)
        """
        x = self.intra_chunk_block(input)
        output = self.inter_chunk_block(x)
        
        return output


class IntraChunkTransformer(nn.Module):
    def __init__(self, num_features, hidden_channels, eps=EPS):
        super().__init__()

        self.num_features, self.hidden_channels = num_features, hidden_channels
        num_directions = 2 # bi-direction
        
        self.rnn = nn.LSTM(num_features, hidden_channels//num_directions, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_channels, num_features)
        self.norm1d = choose_layer_norm(num_features, causal=False, eps=eps)


class InterChunkTransformer(nn.Module):
    def __init__(self, num_features, hidden_channels, causal, eps=EPS):
        super().__init__()
        
        raise NotImplementedError

class ImprovedTransformer(nn.Module):
    def __init__(self, num_features, hidden_channels, num_heads=4, causal=False, eps=EPS):
        super().__init__()

        self.multihead_attn_block = MultiheadAttentionBlock(num_features, num_heads, causal=causal, eps=eps)
        self.subnet = FeedForwardBlock(num_features, hidden_channels, causal=causal, eps=eps)

    def forward(self, input):
        """
        Args:
            input (T, batch_size, num_features)
        Returns:
            output (T, batch_size, num_features)
        """
        x = self.multihead_attn_block(input)
        output = self.subnet(x)
        
        return output

class MultiheadAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, causal=False, eps=EPS):
        super().__init__()

        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm1d = choose_layer_norm(embed_dim, causal=causal, eps=eps)
    
    def forward(self, input):
        """
        Args:
            input (T, batch_size, embed_dim)
        Returns:
            output (T, batch_size, embed_dim)
        """
        x = input # (T, batch_size, embed_dim)

        residual = x
        x, attn_output_weights = self.multihead_attn(x, x, x) # (T_tgt, batch_size, embed_dim), (batch_size, T_tgt, T_src), where T_tgt = T_src = T
        x = x + residual
        x = x.permute(1,2,0) # (batch_size, embed_dim, T)
        x = self.norm1d(x) # (batch_size, embed_dim, T)
        output = x.permute(2,0,1) # (batch_size, embed_dim, T) -> (T, batch_size, embed_dim)

        return output

class FeedForwardBlock(nn.Module):
    def __init__(self, num_features, hidden_channels, causal=False, eps=EPS):
        super().__init__()

        if causal:
            bidirectional = False
            num_directions = 1 # uni-direction
        else:
            bidirectional = True
            num_directions = 2 # bi-direction

        self.rnn = nn.LSTM(num_features, hidden_channels, batch_first=False, bidirectional=bidirectional)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(num_directions*hidden_channels, num_features)
        self.norm1d = choose_layer_norm(num_features, causal=causal, eps=eps)
    
    def forward(self, input):
        """
        Args:
            input (T, batch_size, num_features)
        Returns:
            output (T, batch_size, num_features)
        """
        x = input # (T, batch_size, num_features)

        residual = x
        x, (_, _) = self.rnn(x) # (T, batch_size, num_features) -> (T, batch_size, num_directions*hidden_channels)
        x = self.relu(x) # -> (T, batch_size, num_directions*hidden_channels)
        x = self.fc(x) # (T, batch_size, num_directions*hidden_channels) -> (T, batch_size, num_features)
        x = x + residual
        x = x.permute(1,2,0) # (T, batch_size, num_features) -> (batch_size, num_features, T)
        x = self.norm1d(x) # (batch_size, num_features, T)
        output = x.permute(2,0,1) # (batch_size, num_features, T) -> (T, batch_size, num_features)

        return output

if __name__ == '__main__':
    batch_size = 2
    T = 16

    print('='*10, "Multihead attention block", '='*10)
    embed_dim = 8
    num_heads = 4
    input = torch.rand(T, batch_size, embed_dim)

    print('-'*10, "Non causal", '-'*10)
    causal = False
    
    model = MultiheadAttentionBlock(embed_dim, num_heads=num_heads, causal=causal)
    print(model)

    output = model(input)
    print(input.size(), output.size())
    print()

    print('='*10, "feed-forward block", '='*10)
    num_features = 3
    hidden_channels = 5
    input = torch.rand(T, batch_size, num_features)

    print('-'*10, "Causal", '-'*10)
    causal = True
    
    model = FeedForwardBlock(num_features, hidden_channels, causal=causal)
    print(model)

    output = model(input)
    print(input.size(), output.size())
    print()

    
    print('='*10, "improved transformer", '='*10)
    print('-'*10, "Non causal", '-'*10)
    num_features, hidden_channels = 12, 10
    num_heads = 4
    causal = False
    input = torch.rand(T, batch_size, num_features)

    model = ImprovedTransformer(num_features, hidden_channels, num_heads=num_heads, causal=causal)
    print(model)

    output = model(input)
    print(input.size(), output.size())
    print()

    print('='*10, "Dual path transformer network", '='*10)


    print()