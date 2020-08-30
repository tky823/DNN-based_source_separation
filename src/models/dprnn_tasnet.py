import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils_tasnet import choose_basis, choose_layer_norm
from models.dprnn import DPRNN

EPS=1e-12

class DPRNNTasNet(nn.Module):
    def __init__(self, n_basis, kernel_size, stride=None, enc_basis=None, dec_basis=None, sep_hidden_channels=256, sep_chunk_size=100, sep_hop_size=50, sep_num_blocks=6, dilated=True, separable=True, causal=True, sep_norm=True, eps=EPS, mask_nonlinear='sigmoid', n_sources=2, **kwargs):
        super().__init__()
        
        if stride is None:
            stride = kernel_size//2
        
        assert kernel_size%stride == 0, "kernel_size is expected divisible by stride"
        
        # Encoder-decoder
        self.n_basis = n_basis
        self.kernel_size, self.stride = kernel_size, stride
        self.enc_basis, self.dec_basis = enc_basis, dec_basis
        
        if enc_basis == 'trainable':
            self.enc_nonlinear = kwargs['enc_nonlinear']
        else:
            self.enc_nonlinear = None
        
        if enc_basis in ['Fourier', 'trainableFourier'] or dec_basis in ['Fourier', 'trainableFourier']:
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
        encoder, decoder = choose_basis(n_basis, kernel_size=kernel_size, stride=stride, enc_basis=enc_basis, dec_basis=dec_basis, **kwargs)
        
        self.encoder = encoder
        self.separator = Separator(n_basis, hidden_channels=sep_hidden_channels, chunk_size=sep_chunk_size, hop_size=sep_hop_size, num_blocks=sep_num_blocks, causal=causal, norm=sep_norm, mask_nonlinear=mask_nonlinear, n_sources=n_sources, eps=eps)
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
            latent (batch_size, n_sources, n_basis, T'), where T' = (T-K)//S+1
        """
        n_sources = self.n_sources
        n_basis = self.n_basis
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
        w_hat = w_hat.view(batch_size*n_sources, n_basis, -1)
        x_hat = self.decoder(w_hat)
        x_hat = x_hat.view(batch_size, n_sources, -1)
        output = F.pad(x_hat, (-padding_left, -padding_right))
        
        return output, latent
    
    def get_package(self):
        package = {
            'n_basis': self.n_basis,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'enc_basis': self.enc_basis,
            'dec_basis': self.dec_basis,
            'enc_nonlinear': self.enc_nonlinear,
            'window_fn': self.window_fn,
            'sep_hidden_channels': self.sep_hidden_channels,
            'sep_chunk_size': self.sep_chunk_size,
            'sep_hop_size': self.sep_hop_size,
            'sep_num_blocks': self.sep_num_blocks,
            'dilated': self.dilated,
            'separable': self.separable,
            'causal': self.causal,
            'sep_norm': self.sep_norm,
            'mask_nonlinear': self.mask_nonlinear,
            'n_sources': self.n_sources,
            'eps': self.eps
        }
    
        return package
    
    @classmethod
    def build_model(cls, model_path):
        package = torch.load(model_path, map_location=lambda storage, loc: storage)
        
        n_basis = package['n_basis']
        kernel_size, stride = package['kernel_size'], package['stride']
        enc_basis, dec_basis = package['enc_basis'], package['dec_basis']
        enc_nonlinear = package['enc_nonlinear']
        window_fn = package['window_fn']
        
        sep_hidden_channels = package['sep_hidden_channels']
        sep_chunk_size, sep_hop_size = package['sep_chunk_size'], package['sep_hop_size']
        sep_num_blocks = package['sep_num_blocks']
        
        dilated, separable, causal = package['dilated'], package['separable'], package['causal']
        sep_norm = package['sep_norm']
        mask_nonlinear = package['mask_nonlinear']
        
        n_sources = package['n_sources']
        
        eps = package['eps']
        
        model = cls(n_basis, kernel_size=kernel_size, stride=stride, enc_basis=enc_basis, dec_basis=dec_basis, enc_nonlinear=enc_nonlinear, window_fn=window_fn, sep_hidden_channels=sep_hidden_channels, sep_bottleneck_channels=sep_bottleneck_channels, sep_skip_channels=sep_skip_channels, sep_kernel_size=sep_kernel_size, sep_num_blocks=sep_num_blocks, sep_num_layers=sep_num_layers, dilated=dilated, separable=separable, causal=causal, sep_nonlinear=sep_nonlinear, sep_norm=sep_norm, mask_nonlinear=mask_nonlinear, n_sources=n_sources, eps=eps)
        
        return model
    
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
        
        self.norm1d = choose_layer_norm(num_features, causal=causal, eps=eps)
        self.segment1d = Segment1d(chunk_size, hop_size)
        self.dprnn = DPRNN(num_features, hidden_channels, num_blocks=num_blocks, causal=causal)
        self.overlap_add1d = OverlapAdd1d(chunk_size, hop_size)
        
        self.prelu = nn.PReLU()
        self.mask_conv1d = nn.Conv1d(num_features, n_sources*num_features, kernel_size=1, stride=1)
        
        if mask_nonlinear == 'sigmoid':
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
        x = self.prelu(x)
        x = self.mask_conv1d(x)
        x = self.mask_nonlinear(x)
        output = x.view(batch_size, n_sources, num_features, T_bin)
        
        return output

class Segment1d(nn.Module):
    """
    Segmentation. Input tensor is 3-D (audio-like), but output tensor is 4-D (image-like).
    """
    def __init__(self, chunk_size, hop_size):
        super().__init__()
        
        self.chunk_size, self.hop_size = chunk_size, hop_size

    def forward(self, input):
        """
        Args:
            input (batch_size, num_features, T_bin)
        Returns:
            output (batch_size, num_features, S, chunk_size): S is length of global output, where S = (T_bin-chunk_size)//hop_size + 1
        """
        chunk_size, hop_size = self.chunk_size, self.hop_size
        batch_size, num_features, T_bin = input.size()
        
        input = input.view(batch_size,num_features,T_bin,1)
        x = F.unfold(input, kernel_size=(chunk_size,1), stride=(hop_size,1)) # -> (batch_size, num_features*chunk_size, S), where S = (T_bin-chunk_size)//hop_size+1
        x = x.view(batch_size, num_features, chunk_size, -1)
        output = x.permute(0,1,3,2).contiguous() # -> (batch_size, num_features, S, chunk_size)
        
        return output


class OverlapAdd1d(nn.Module):
    """
    Overlap-add operation. Input tensor is 4-D (image-like), but output tensor is 3-D (audio-like).
    """
    def __init__(self, chunk_size, hop_size):
        super().__init__()
        
        self.chunk_size, self.hop_size = chunk_size, hop_size
        
    def forward(self, input):
        """
        Args:
            input: (batch_size, num_features, S, chunk_size)
        Returns:
            output: (batch_size, num_features, T_bin)
        """
        chunk_size, hop_size = self.chunk_size, self.hop_size
        batch_size, num_features, S, chunk_size = input.size()
        T_bin = (S - 1) * hop_size + chunk_size
        
        x = input.permute(0,1,3,2).contiguous() # -> (batch_size, num_features, chunk_size, S)
        x = x.view(batch_size, num_features*chunk_size, S) # -> (batch_size, num_features*chunk_size, S)
        output = F.fold(x, kernel_size=(chunk_size,1), stride=(hop_size,1), output_size=(T_bin,1)) # -> (batch_size, num_features, T_bin, 1)
        output = output.squeeze(dim=3)
        
        return output

if __name__ == '__main__':
    batch_size, num_features, T_bin = 2, 3, 5
    K, P = 3, 2
    S = (T_bin-K)//P + 1
    
    print("="*10, "Segment", "="*10)
    input = torch.randint(0, 10, (batch_size, num_features, T_bin), dtype=torch.float)
    
    segment = Segment1d(K, hop_size=P)
    output = segment(input)
    
    print(input.size(), output.size())
    print(input)
    print(output)
    print()
    
    print("="*10, "OverlapAdd", "="*10)
    input = torch.randint(0, 10, (batch_size, num_features, S, K), dtype=torch.float)
    
    overlap_add = OverlapAdd1d(K, hop_size=P)
    output = overlap_add(input)
    
    print(input.size(), output.size())
    print(input)
    print(output)
    print()
    
    print("="*10, "Separator", "="*10)
    N, H = 16, 32
    B = 3
    
    sep_norm = True
    mask_nonlinear = 'sigmoid'
    
    causal = True
    n_sources = 2
    
    input = torch.randint(0, 10, (batch_size, N, T_bin), dtype=torch.float)
    
    separator = Separator(N, hidden_channels=H, chunk_size=K, hop_size=P, num_blocks=B, causal=causal, norm=sep_norm, mask_nonlinear=mask_nonlinear, n_sources=n_sources)
    
    output = separator(input)
    print(input.size(), output.size())
    print()
    
    print("="*10, "DPRNN-TasNet", "="*10)
    # Encoder & decoder
    C, T = 1, 128
    L, N = 8, 16
    
    # Separator
    H = 32
    B = 4
    dilated, separable, sep_norm = True, True, True
    
    input = torch.randn((batch_size, C, T), dtype=torch.float)
    
    print("-"*10, "Trainable Basis & Non causal", "-"*10)
    enc_basis, dec_basis = 'trainable', 'trainable'
    enc_nonlinear = 'relu'
    
    causal = False
    mask_nonlinear = 'sigmoid'
    n_sources = 2
    
    model = DPRNNTasNet(N, kernel_size=L, enc_basis=enc_basis, dec_basis=dec_basis, enc_nonlinear=enc_nonlinear, sep_hidden_channels=H, sep_chunk_size=K, sep_hop_size=P, sep_num_blocks=B, dilated=dilated, separable=separable, causal=causal, sep_norm=sep_norm, mask_nonlinear=mask_nonlinear, n_sources=n_sources)
    print(model)
    print("# Parameters: {}".format(model.num_parameters))
    
    output = model(input)
    print(input.size(), output.size())
    print()
    
    print("-"*10, "Fourier Basis & Causal", "-"*10)
    enc_basis, dec_basis = 'Fourier', 'Fourier'
    window_fn = 'hamming'
    
    causal = True
    mask_nonlinear = 'softmax'
    n_sources = 3
    
    model = DPRNNTasNet(N, kernel_size=L, enc_basis=enc_basis, dec_basis=dec_basis, window_fn=window_fn, sep_hidden_channels=H, sep_chunk_size=K, sep_hop_size=P, sep_num_blocks=B, dilated=dilated, separable=separable, causal=causal, sep_norm=sep_norm, mask_nonlinear=mask_nonlinear, n_sources=n_sources)
    print(model)
    print("# Parameters: {}".format(model.num_parameters))
    
    output = model(input)
    print(input.size(), output.size())
    print()
