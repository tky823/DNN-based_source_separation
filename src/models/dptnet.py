import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils_filterbank import choose_filterbank
from utils.utils_tasnet import choose_layer_norm
from models.gtu import GTU1d
from models.dprnn_tasnet import Segment1d, OverlapAdd1d
from models.dptransformer import DualPathTransformer

EPS=1e-12

class DPTNet(nn.Module):
    """
    Dual-path transformer based network
    """
    def __init__(
        self,
        n_basis, kernel_size, stride=None,
        enc_basis=None, dec_basis=None,
        sep_bottleneck_channels=64, sep_hidden_channels=256,
        sep_chunk_size=100, sep_hop_size=None, sep_num_blocks=6,
        sep_num_heads=4, sep_norm=True, sep_nonlinear='relu', sep_dropout=0,
        mask_nonlinear='relu',
        causal=False,
        n_sources=2,
        eps=EPS,
        **kwargs
    ):
        super().__init__()
        
        if stride is None:
            stride = kernel_size // 2
        
        if sep_hop_size is None:
            sep_hop_size = sep_chunk_size // 2
        
        assert kernel_size % stride == 0, "kernel_size is expected divisible by stride"
        assert n_basis % sep_num_heads == 0, "n_basis must be divisible by sep_num_heads"
        
        # Encoder-decoder
        self.n_basis = n_basis
        self.kernel_size, self.stride = kernel_size, stride
        self.enc_basis, self.dec_basis = enc_basis, dec_basis
        
        if enc_basis == 'trainable' and not dec_basis == 'pinv':    
            self.enc_nonlinear = kwargs['enc_nonlinear']
        else:
            self.enc_nonlinear = None
        
        if enc_basis in ['Fourier', 'trainableFourier', 'trainableFourierTrainablePhase'] or dec_basis in ['Fourier', 'trainableFourier', 'trainableFourierTrainablePhase']:
            self.window_fn = kwargs['window_fn']
            self.enc_onesided, self.enc_return_complex = kwargs['enc_onesided'], kwargs['enc_return_complex']
        else:
            self.window_fn = None
            self.enc_onesided, self.enc_return_complex = None, None
        
        # Separator configuration
        self.sep_bottleneck_channels, self.sep_hidden_channels = sep_bottleneck_channels, sep_hidden_channels
        self.sep_chunk_size, self.sep_hop_size = sep_chunk_size, sep_hop_size
        self.sep_num_blocks = sep_num_blocks
        self.sep_num_heads = sep_num_heads
        self.sep_norm = sep_norm
        self.sep_nonlinear = sep_nonlinear
        self.sep_dropout = sep_dropout

        self.causal = causal
        self.mask_nonlinear = mask_nonlinear
        
        self.n_sources = n_sources
        self.eps = eps
        
        # Network configuration
        encoder, decoder = choose_filterbank(n_basis, kernel_size=kernel_size, stride=stride, enc_basis=enc_basis, dec_basis=dec_basis, **kwargs)
        
        self.encoder = encoder
        self.separator = Separator(
            n_basis, bottleneck_channels=sep_bottleneck_channels, hidden_channels=sep_hidden_channels,
            chunk_size=sep_chunk_size, hop_size=sep_hop_size, num_blocks=sep_num_blocks,
            num_heads=sep_num_heads, norm=sep_norm, nonlinear=sep_nonlinear, dropout=sep_dropout,
            mask_nonlinear=mask_nonlinear,
            causal=causal,
            n_sources=n_sources,
            eps=eps
        )
        self.decoder = decoder
        
    def forward(self, input):
        output, _ = self.extract_latent(input)
        
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
        
        assert C_in == 1, "input.size() is expected (?, 1, ?), but given {}".format(input.size())
        
        padding = (stride - (T - kernel_size) % stride) % stride
        padding_left = padding // 2
        padding_right = padding - padding_left

        input = F.pad(input, (padding_left, padding_right))
        w = self.encoder(input)

        if torch.is_complex(w):
            amplitude, phase = torch.abs(w), torch.angle(w)
            mask = self.separator(amplitude)
            amplitude, phase = amplitude.unsqueeze(dim=1), phase.unsqueeze(dim=1)
            w_hat = amplitude * mask * torch.exp(1j * phase)
        else:
            mask = self.separator(w)
            w = w.unsqueeze(dim=1)
            w_hat = w * mask
        
        latent = w_hat
        w_hat = w_hat.view(batch_size*n_sources, n_basis, -1)
        x_hat = self.decoder(w_hat)
        x_hat = x_hat.view(batch_size, n_sources, -1)
        output = F.pad(x_hat, (-padding_left, -padding_right))
        
        return output, latent
    
    def get_config(self):
        config = {
            'n_basis': self.n_basis,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'enc_basis': self.enc_basis,
            'dec_basis': self.dec_basis,
            'enc_nonlinear': self.enc_nonlinear,
            'window_fn': self.window_fn,
            'enc_onesided': self.enc_onesided,
            'enc_return_complex': self.enc_return_complex,
            'sep_hidden_channels': self.sep_hidden_channels,
            'sep_bottleneck_channels': self.sep_bottleneck_channels,
            'sep_chunk_size': self.sep_chunk_size,
            'sep_hop_size': self.sep_hop_size,
            'sep_num_blocks': self.sep_num_blocks,
            'sep_num_heads': self.sep_num_heads,
            'sep_norm': self.sep_norm,
            'sep_nonlinear': self.sep_nonlinear,
            'sep_dropout': self.sep_dropout,
            'mask_nonlinear': self.mask_nonlinear,
            'causal': self.causal,
            'n_sources': self.n_sources,
            'eps': self.eps
        }
    
        return config
    
    @classmethod
    def build_model(cls, model_path, load_state_dict=False):
        config = torch.load(model_path, map_location=lambda storage, loc: storage)
        
        n_basis = config.get('n_bases') or config['n_basis']
        kernel_size, stride = config['kernel_size'], config['stride']
        enc_basis, dec_basis = config.get('enc_bases') or config['enc_basis'], config.get('dec_bases') or config['dec_basis']
        enc_nonlinear = config['enc_nonlinear']
        enc_onesided, enc_return_complex = config.get('enc_onesided') or None, config.get('enc_return_complex') or None
        window_fn = config['window_fn']
        
        sep_hidden_channels, sep_bottleneck_channels = config['sep_hidden_channels'], config['sep_bottleneck_channels']
        sep_chunk_size, sep_hop_size = config['sep_chunk_size'], config['sep_hop_size']
        sep_num_blocks = config['sep_num_blocks']
        sep_num_heads = config['sep_num_heads']
        sep_norm, sep_nonlinear, sep_dropout = config['sep_norm'], config['sep_nonlinear'], config['sep_dropout']
        
        sep_nonlinear, sep_norm = config['sep_nonlinear'], config['sep_norm']
        mask_nonlinear = config['mask_nonlinear']

        causal = config['causal']
        n_sources = config['n_sources']
        
        eps = config['eps']

        model = cls(
            n_basis, kernel_size, stride=stride,
            enc_basis=enc_basis, dec_basis=dec_basis, enc_nonlinear=enc_nonlinear,
            window_fn=window_fn, enc_onesided=enc_onesided, enc_return_complex=enc_return_complex,
            sep_bottleneck_channels=sep_bottleneck_channels, sep_hidden_channels=sep_hidden_channels,
            sep_chunk_size=sep_chunk_size, sep_hop_size=sep_hop_size, sep_num_blocks=sep_num_blocks,
            sep_num_heads=sep_num_heads,
            sep_norm=sep_norm, sep_nonlinear=sep_nonlinear, sep_dropout=sep_dropout,
            mask_nonlinear=mask_nonlinear,
            causal=causal,
            n_sources=n_sources,
            eps=eps
        )
        
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

class Separator(nn.Module):
    def __init__(
        self,
        num_features, bottleneck_channels=32, hidden_channels=128,
        chunk_size=100, hop_size=None, num_blocks=6,
        num_heads=4,
        norm=True, nonlinear='relu', dropout=0,
        mask_nonlinear='relu',
        causal=True,
        n_sources=2,
        eps=EPS
    ):
        super().__init__()

        if hop_size is None:
            hop_size = chunk_size//2
        
        self.num_features, self.n_sources = num_features, n_sources
        self.chunk_size, self.hop_size = chunk_size, hop_size
        
        self.bottleneck_conv1d = nn.Conv1d(num_features, bottleneck_channels, kernel_size=1, stride=1)
        self.segment1d = Segment1d(chunk_size, hop_size)
        
        norm_name = 'cLN' if causal else 'gLN'
        self.norm2d = choose_layer_norm(norm_name, bottleneck_channels, causal=causal, eps=eps)

        self.dptransformer = DualPathTransformer(
            bottleneck_channels, hidden_channels,
            num_blocks=num_blocks, num_heads=num_heads,
            norm=norm, nonlinear=nonlinear, dropout=dropout,
            causal=causal, eps=eps
        )
        self.overlap_add1d = OverlapAdd1d(chunk_size, hop_size)
        self.prelu = nn.PReLU()
        self.map = nn.Conv1d(bottleneck_channels, n_sources*num_features, kernel_size=1, stride=1)
        self.gtu = GTU1d(num_features, num_features, kernel_size=1, stride=1)
        
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
            input (batch_size, num_features, n_frames)
        Returns:
            output (batch_size, n_sources, num_features, n_frames)
        """
        num_features, n_sources = self.num_features, self.n_sources
        chunk_size, hop_size = self.chunk_size, self.hop_size
        batch_size, num_features, n_frames = input.size()
        
        padding = (hop_size - (n_frames - chunk_size) % hop_size) % hop_size
        padding_left = padding // 2
        padding_right = padding - padding_left
        
        x = self.bottleneck_conv1d(input)
        x = F.pad(x, (padding_left, padding_right))
        x = self.segment1d(x) # -> (batch_size, C, S, chunk_size)
        x = self.norm2d(x)
        x = self.dptransformer(x)
        x = self.overlap_add1d(x)
        x = F.pad(x, (-padding_left, -padding_right))
        x = self.prelu(x) # -> (batch_size, C, n_frames)
        x = self.map(x) # -> (batch_size, n_sources*C, n_frames)
        x = x.view(batch_size*n_sources, num_features, n_frames) # -> (batch_size*n_sources, num_features, n_frames)
        x = self.gtu(x) # -> (batch_size*n_sources, num_features, n_frames)
        x = self.mask_nonlinear(x) # -> (batch_size*n_sources, num_features, n_frames)
        output = x.view(batch_size, n_sources, num_features, n_frames)
        
        return output

def _test_separator():
    batch_size = 2
    T_bin = 64
    n_sources = 3

    num_features = 10
    d = 12 # must be divisible by num_heads
    d_ff = 15
    chunk_size = 10 # local chunk length
    num_blocks = 3
    num_heads = 4 # multihead attention in transformer

    input = torch.randn((batch_size, num_features, T_bin), dtype=torch.float)
    
    causal = False

    separator = Separator(
        num_features, hidden_channels=d_ff, bottleneck_channels=d,
        chunk_size=chunk_size, num_blocks=num_blocks, num_heads=num_heads,
        causal=causal,
        n_sources=n_sources
    )
    print(separator)

    output = separator(input)
    print(input.size(), output.size())

def _test_dptnet():
    batch_size = 2
    T = 64

    # Encoder decoder
    N, L = 12, 8
    enc_basis, dec_basis = 'trainable', 'trainable'
    enc_nonlinear = 'relu'
    
    # Separator
    d = 32 # must be divisible by num_heads
    d_ff = 4 * d # depth of feed-forward network
    K = 10 # local chunk length
    B, h = 3, 4 # number of dual path transformer processing block, and multihead attention in transformer
    mask_nonlinear = 'relu'
    n_sources = 2

    input = torch.randn((batch_size, 1, T), dtype=torch.float)
    
    causal = False

    model = DPTNet(
        N, L, enc_basis=enc_basis, dec_basis=dec_basis, enc_nonlinear=enc_nonlinear,
        sep_bottleneck_channels=d, sep_hidden_channels=d_ff,
        sep_chunk_size=K, sep_num_blocks=B, sep_num_heads=h,
        mask_nonlinear=mask_nonlinear,
        causal=causal,
        n_sources=n_sources
    )
    print(model)

    output = model(input)
    print("# Parameters: {}".format(model.num_parameters))
    print(input.size(), output.size())

def _test_dptnet_paper():
    batch_size = 2
    T = 64

    # Encoder decoder
    N, L = 64, 2
    enc_basis, dec_basis = 'trainable', 'trainable'
    enc_nonlinear = 'relu'
    
    # Separator
    d = 32
    d_ff = 4 * d # depth of feed-forward network
    K = 10 # local chunk length
    B, h = 6, 4 # number of dual path transformer processing block, and multihead attention in transformer
    
    mask_nonlinear = 'relu'
    n_sources = 2

    input = torch.randn((batch_size, 1, T), dtype=torch.float)
    
    causal = False

    model = DPTNet(
        N, L, enc_basis=enc_basis, dec_basis=dec_basis, enc_nonlinear=enc_nonlinear,
        sep_bottleneck_channels=N, sep_hidden_channels=d_ff,
        sep_chunk_size=K, sep_num_blocks=B, sep_num_heads=h,
        mask_nonlinear=mask_nonlinear,
        causal=causal,
        n_sources=n_sources
    )
    print(model)

    output = model(input)
    print("# Parameters: {}".format(model.num_parameters))
    print(input.size(), output.size())

if __name__ == '__main__':
    print('='*10, "Separator based on dual path transformer network", '='*10)
    _test_separator()
    print()

    print('='*10, "Dual path transformer network", '='*10)
    _test_dptnet()
    print()

    print('='*10, "Dual path transformer network (same configuration in the paper)", '='*10)
    _test_dptnet_paper()
    print()