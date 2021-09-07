import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils_tasnet import choose_bases, choose_layer_norm
from models.transform import Segment1d, OverlapAdd1d
from models.dprnn import DPRNN

EPS=1e-12

class DPRNNTasNet(nn.Module):
    def __init__(
        self,
        n_bases, kernel_size, stride=None, enc_bases=None, dec_bases=None,
        sep_hidden_channels=128, sep_bottleneck_channels=64,
        sep_chunk_size=100, sep_hop_size=50,
        sep_num_blocks=6,
        sep_norm=True, mask_nonlinear='sigmoid',
        causal=True,
        n_sources=2,
        eps=EPS,
        **kwargs
    ):
        super().__init__()
        
        if stride is None:
            stride = kernel_size//2
        
        assert kernel_size%stride == 0, "kernel_size is expected divisible by stride"
        
        # Encoder-decoder
        if 'in_channels' in kwargs:
            self.in_channels = kwargs['in_channels']
        else:
            self.in_channels = 1
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
        self.sep_hidden_channels, self.sep_bottleneck_channels = sep_hidden_channels, sep_bottleneck_channels
        self.sep_chunk_size, self.sep_hop_size = sep_chunk_size, sep_hop_size
        self.sep_num_blocks = sep_num_blocks
        
        self.causal = causal
        self.sep_norm = sep_norm
        self.mask_nonlinear = mask_nonlinear
        
        self.n_sources = n_sources
        self.eps = eps
        
        # Network configuration
        encoder, decoder = choose_bases(n_bases, kernel_size=kernel_size, stride=stride, enc_bases=enc_bases, dec_bases=dec_bases, **kwargs)
        
        self.encoder = encoder
        self.separator = Separator(
            n_bases, bottleneck_channels=sep_bottleneck_channels, hidden_channels=sep_hidden_channels,
            chunk_size=sep_chunk_size, hop_size=sep_hop_size,
            num_blocks=sep_num_blocks, norm=sep_norm, mask_nonlinear=mask_nonlinear,
            causal=causal,
            n_sources=n_sources,
            eps=eps
        )
        self.decoder = decoder
        
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
        
        n_dim = input.dim()

        if n_dim == 3:
            batch_size, C_in, T = input.size()
            assert C_in == 1, "input.size() is expected (?,1,?), but given {}".format(input.size())
        elif n_dim == 4:
            batch_size, C_in, n_mics, T = input.size()
            assert C_in == 1, "input.size() is expected (?,1,?,?), but given {}".format(input.size())
            input = input.view(batch_size, n_mics, T)
        else:
            raise ValueError("Not support {} dimension input".format(n_dim))
        
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
        if n_dim == 3:
            x_hat = x_hat.view(batch_size, n_sources, -1)
        else: # n_dim == 4
            x_hat = x_hat.view(batch_size, n_sources, n_mics, -1)
        output = F.pad(x_hat, (-padding_left, -padding_right))
        
        return output, latent
    
    def get_package(self):
        package = {
            'in_channels': self.in_channels,
            'n_bases': self.n_bases,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'enc_bases': self.enc_bases,
            'dec_bases': self.dec_bases,
            'enc_nonlinear': self.enc_nonlinear,
            'window_fn': self.window_fn,
            'sep_hidden_channels': self.sep_hidden_channels,
            'sep_bottleneck_channels': self.sep_bottleneck_channels,
            'sep_chunk_size': self.sep_chunk_size,
            'sep_hop_size': self.sep_hop_size,
            'sep_num_blocks': self.sep_num_blocks,
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
        
        in_channels = package.get('in_channels') or 1
        n_bases = package['n_bases']
        kernel_size, stride = package['kernel_size'], package['stride']
        enc_bases, dec_bases = package['enc_bases'], package['dec_bases']
        enc_nonlinear = package['enc_nonlinear']
        window_fn = package['window_fn']
        
        sep_hidden_channels, sep_bottleneck_channels = package['sep_hidden_channels'], package['sep_bottleneck_channels']
        sep_chunk_size, sep_hop_size = package['sep_chunk_size'], package['sep_hop_size']
        sep_num_blocks = package['sep_num_blocks']
        
        sep_norm = package['sep_norm']
        mask_nonlinear = package['mask_nonlinear']

        causal = package['causal']
        n_sources = package['n_sources']
        
        eps = package['eps']
        
        model = cls(
            n_bases, in_channels=in_channels, kernel_size=kernel_size, stride=stride, enc_bases=enc_bases, dec_bases=dec_bases, enc_nonlinear=enc_nonlinear, window_fn=window_fn,
            sep_hidden_channels=sep_hidden_channels, sep_bottleneck_channels=sep_bottleneck_channels,
            sep_chunk_size=sep_chunk_size, sep_hop_size=sep_hop_size,
            sep_num_blocks=sep_num_blocks,
            sep_norm=sep_norm, mask_nonlinear=mask_nonlinear,
            causal=causal,
            n_sources=n_sources,
            eps=eps
        )
        
        return model
    
    @property
    def num_parameters(self):
        _num_parameters = 0
        
        for p in self.parameters():
            if p.requires_grad:
                _num_parameters += p.numel()
                
        return _num_parameters

class Separator(nn.Module):
    def __init__(
        self,
        num_features, bottleneck_channels=64, hidden_channels=128,
        chunk_size=100, hop_size=50,
        num_blocks=6,
        norm=True, mask_nonlinear='sigmoid',
        causal=True,
        n_sources=2,
        eps=EPS
    ):
        super().__init__()
        
        self.num_features, self.n_sources = num_features, n_sources
        self.chunk_size, self.hop_size = chunk_size, hop_size
        self.norm = norm
        
        norm_name = 'cLN' if causal else 'gLM'
        self.norm1d = choose_layer_norm(norm_name, num_features, causal=causal, eps=eps)
        self.bottleneck_conv1d = nn.Conv1d(num_features, bottleneck_channels, kernel_size=1, stride=1)
        
        self.segment1d = Segment1d(chunk_size, hop_size)
        self.dprnn = DPRNN(bottleneck_channels, hidden_channels, num_blocks=num_blocks, causal=causal, norm=norm)
        self.overlap_add1d = OverlapAdd1d(chunk_size, hop_size)
        
        self.prelu = nn.PReLU()
        self.mask_conv1d = nn.Conv1d(bottleneck_channels, n_sources*num_features, kernel_size=1, stride=1)
        
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
        
        x = self.norm1d(input)
        x = self.bottleneck_conv1d(x)
        x = F.pad(x, (padding_left, padding_right))
        x = self.segment1d(x)
        x = self.dprnn(x)
        x = self.overlap_add1d(x)
        x = F.pad(x, (-padding_left, -padding_right))
        x = self.prelu(x)
        x = self.mask_conv1d(x)
        x = self.mask_nonlinear(x)
        output = x.view(batch_size, n_sources, num_features, T_bin)
        
        return output

def _test_separator():
    batch_size, T_bin = 2, 5
    N, F, H = 16, 16, 32 # H is the number of channels for each direction
    K, P = 3, 2
    B = 3
    
    sep_norm = True
    mask_nonlinear = 'sigmoid'
    
    causal = True
    n_sources = 2
    
    input = torch.randn((batch_size, N, T_bin), dtype=torch.float)
    
    separator = Separator(
        N, bottleneck_channels=F, hidden_channels=H,
        chunk_size=K, hop_size=P,
        num_blocks=B,
        norm=sep_norm, mask_nonlinear=mask_nonlinear,
        causal=causal,
        n_sources=n_sources
    )
    print(separator)

    output = separator(input)
    print(input.size(), output.size())

def _test_dprnn_tasnet():
    batch_size, T_bin = 2, 5
    K, P = 3, 2

    # Encoder & decoder
    C, T = 1, 128
    L, N = 8, 16
    
    # Separator
    F = N
    H = 32 # for each direction
    B = 4
    sep_norm = True
    
    input = torch.randn((batch_size, C, T), dtype=torch.float)
    
    print("-"*10, "Trainable Bases & Non causal", "-"*10)
    enc_bases, dec_bases = 'trainable', 'trainable'
    enc_nonlinear = 'relu'
    
    causal = False
    mask_nonlinear = 'sigmoid'
    n_sources = 2
    
    model = DPRNNTasNet(
        N, kernel_size=L, enc_bases=enc_bases, dec_bases=dec_bases, enc_nonlinear=enc_nonlinear,
        sep_hidden_channels=H, sep_bottleneck_channels=F,
        sep_chunk_size=K, sep_hop_size=P,
        sep_num_blocks=B,
        sep_norm=sep_norm, mask_nonlinear=mask_nonlinear,
        causal=causal,
        n_sources=n_sources
    )
    print(model)
    print("# Parameters: {}".format(model.num_parameters))
    
    output = model(input)
    print(input.size(), output.size())
    print()
    
    print("-"*10, "Fourier Bases & Causal", "-"*10)
    enc_bases, dec_bases = 'Fourier', 'Fourier'
    window_fn = 'hamming'
    
    causal = True
    mask_nonlinear = 'softmax'
    n_sources = 3
    
    model = DPRNNTasNet(
        N, kernel_size=L, enc_bases=enc_bases, dec_bases=dec_bases, window_fn=window_fn,
        sep_hidden_channels=H, sep_bottleneck_channels=F,
        sep_chunk_size=K, sep_hop_size=P,
        sep_num_blocks=B, sep_norm=sep_norm, mask_nonlinear=mask_nonlinear,
        causal=causal,
        n_sources=n_sources
    )
    print(model)
    print("# Parameters: {}".format(model.num_parameters))
    
    output = model(input)
    print(input.size(), output.size())


def _test_multichannel_dprnn_tasnet():
    batch_size = 4
    K, P = 3, 2

    # Encoder & decoder
    C, T = 2, 128
    L, N = 8, 16
    
    # Separator
    F = N
    H = 32 # for each direction
    B = 4
    sep_norm = True
    
    input = torch.randn((batch_size, 1, C, T), dtype=torch.float)
    
    enc_bases, dec_bases = 'trainable', 'trainable'
    enc_nonlinear = 'relu'
    
    causal = False
    mask_nonlinear = 'sigmoid'
    n_sources = 3
    
    model = DPRNNTasNet(
        N, in_channels=C, kernel_size=L, enc_bases=enc_bases, dec_bases=dec_bases, enc_nonlinear=enc_nonlinear,
        sep_hidden_channels=H, sep_bottleneck_channels=F,
        sep_chunk_size=K, sep_hop_size=P,
        sep_num_blocks=B,
        sep_norm=sep_norm, mask_nonlinear=mask_nonlinear,
        causal=causal,
        n_sources=n_sources
    )
    print(model)
    print("# Parameters: {}".format(model.num_parameters))
    
    output = model(input)
    print(input.size(), output.size())

def _test_dprnn_tasnet_paper():
    print("Only K and P is different from original, but it doesn't affect the # parameters.")
    batch_size = 2
    K, P = 3, 2
    
    # Encoder & decoder
    C, T = 1, 128
    L, N = 2, 64
    
    # Separator
    F = N
    H = 128 # for each direction
    B = 6
    sep_norm = True
    
    input = torch.randn((batch_size, C, T), dtype=torch.float)
    
    print("-"*10, "Trainable Bases & Non causal", "-"*10)
    enc_bases, dec_bases = 'trainable', 'trainable'
    enc_nonlinear = None
    
    causal = False
    mask_nonlinear = 'sigmoid'
    n_sources = 2
    
    model = DPRNNTasNet(
        N, kernel_size=L, enc_bases=enc_bases, dec_bases=dec_bases, enc_nonlinear=enc_nonlinear,
        sep_hidden_channels=H, sep_bottleneck_channels=F,
        sep_num_blocks=B,
        sep_chunk_size=K, sep_hop_size=P,
        sep_norm=sep_norm, mask_nonlinear=mask_nonlinear,
        causal=causal,
        n_sources=n_sources
    )
    print(model)
    print("# Parameters: {}".format(model.num_parameters))
    
    output = model(input)
    print(input.size(), output.size())

if __name__ == '__main__':
    print("="*10, "Separator", "="*10)
    _test_separator()
    print()
    
    print("="*10, "DPRNN-TasNet", "="*10)
    _test_dprnn_tasnet()
    print()

    print("="*10, "DPRNN-TasNet (multichannel)", "="*10)
    _test_multichannel_dprnn_tasnet()
    print()

    print("="*10, "DPRNN-TasNet (same configuration in paper)", "="*10)
    _test_dprnn_tasnet_paper()
    print()