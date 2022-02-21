import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.filterbank import choose_filterbank
from utils.tasnet import choose_layer_norm
from models.gtu import GTU1d
from models.transform import Segment1d, OverlapAdd1d
from models.galr import GALR

EPS = 1e-12

class GALRNet(nn.Module):
    def __init__(
        self,
        n_basis, kernel_size, stride=None, enc_basis=None, dec_basis=None,
        sep_hidden_channels=128,
        sep_chunk_size=100, sep_hop_size=50, sep_down_chunk_size=None, sep_num_blocks=6,
        sep_num_heads=8, sep_norm=True, sep_dropout=0.1,
        mask_nonlinear='relu',
        causal=True,
        n_sources=2,
        low_dimension=True,
        eps=EPS,
        **kwargs
    ):
        super().__init__()

        if stride is None:
            stride = kernel_size // 2

        assert kernel_size % stride == 0, "kernel_size is expected divisible by stride"

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
        self.sep_hidden_channels = sep_hidden_channels
        self.sep_chunk_size, self.sep_hop_size, self.sep_down_chunk_size = sep_chunk_size, sep_hop_size, sep_down_chunk_size
        self.sep_num_blocks = sep_num_blocks
        self.sep_num_heads = sep_num_heads
        self.sep_norm = sep_norm
        self.sep_dropout = sep_dropout
        self.low_dimension = low_dimension

        self.causal = causal
        self.sep_norm = sep_norm
        self.mask_nonlinear = mask_nonlinear

        self.n_sources = n_sources
        self.eps = eps

        # Network configuration
        encoder, decoder = choose_filterbank(n_basis, kernel_size=kernel_size, stride=stride, enc_basis=enc_basis, dec_basis=dec_basis, **kwargs)

        self.encoder = encoder
        self.separator = Separator(
            n_basis, hidden_channels=sep_hidden_channels,
            chunk_size=sep_chunk_size, hop_size=sep_hop_size, down_chunk_size=sep_down_chunk_size, num_blocks=sep_num_blocks,
            num_heads=sep_num_heads, norm=sep_norm, dropout=sep_dropout, mask_nonlinear=mask_nonlinear,
            low_dimension=low_dimension,
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
            'sep_chunk_size': self.sep_chunk_size,
            'sep_hop_size': self.sep_hop_size,
            'sep_down_chunk_size': self.sep_down_chunk_size,
            'sep_num_blocks': self.sep_num_blocks,
            'sep_num_heads': self.sep_num_heads,
            'sep_norm': self.sep_norm,
            'sep_dropout': self.sep_dropout,
            'low_dimension': self.low_dimension,
            'mask_nonlinear': self.mask_nonlinear,
            'causal': self.causal,
            'n_sources': self.n_sources,
            'eps': self.eps
        }

        return config

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
        num_features, hidden_channels=128,
        chunk_size=100, hop_size=50, down_chunk_size=None, num_blocks=6, num_heads=4,
        norm=True, dropout=0.1, mask_nonlinear='relu',
        low_dimension=True,
        causal=True,
        n_sources=2,
        eps=EPS
    ):
        super().__init__()

        self.num_features, self.n_sources = num_features, n_sources
        self.chunk_size, self.hop_size = chunk_size, hop_size

        self.segment1d = Segment1d(chunk_size, hop_size)
        norm_name = 'cLN' if causal else 'gLN'
        self.norm2d = choose_layer_norm(norm_name, num_features, causal=causal, eps=eps)

        if low_dimension:
            # If low-dimension representation, latent_dim and chunk_size are required
            if down_chunk_size is None:
                raise ValueError("Specify down_chunk_size")
            self.galr = GALR(
                num_features, hidden_channels,
                chunk_size=chunk_size, down_chunk_size=down_chunk_size,
                num_blocks=num_blocks, num_heads=num_heads,
                norm=norm, dropout=dropout,
                low_dimension=low_dimension,
                causal=causal,
                eps=eps
            )
        else:
            self.galr = GALR(
                num_features, hidden_channels,
                num_blocks=num_blocks, num_heads=num_heads,
                norm=norm, dropout=dropout,
                low_dimension=low_dimension,
                causal=causal,
                eps=eps
            )
        self.overlap_add1d = OverlapAdd1d(chunk_size, hop_size)
        self.prelu = nn.PReLU()
        self.map = nn.Conv1d(num_features, n_sources*num_features, kernel_size=1, stride=1)
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

        padding = (hop_size-(n_frames-chunk_size)%hop_size)%hop_size
        padding_left = padding//2
        padding_right = padding - padding_left

        x = F.pad(input, (padding_left, padding_right))
        x = self.segment1d(x) # -> (batch_size, C, S, chunk_size)
        x = self.norm2d(x)
        x = self.galr(x)
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
    batch_size, n_frames = 2, 5
    M, H = 16, 32 # H is the number of channels for each direction
    K, P, Q = 3, 2, 2
    N = 3

    sep_norm = True
    mask_nonlinear = 'sigmoid'
    low_dimension = True

    causal = True
    n_sources = 2

    input = torch.randn((batch_size, M, n_frames), dtype=torch.float)

    separator = Separator(
        M, hidden_channels=H,
        chunk_size=K, hop_size=P, down_chunk_size=Q,
        num_blocks=N,
        norm=sep_norm, mask_nonlinear=mask_nonlinear,
        low_dimension=low_dimension,
        causal=causal,
        n_sources=n_sources
    )
    print(separator)

    output = separator(input)
    print(input.size(), output.size())

def _test_galrnet():
    batch_size = 2
    C, T = 1, 128
    K, P, Q = 3, 2, 2

    # Encoder & decoder
    M, D = 8, 16

    # Separator
    H = 32 # for each direction
    N, J = 4, 4
    sep_norm = True

    low_dimension=True

    input = torch.randn((batch_size, C, T), dtype=torch.float)

    print("-"*10, "Trainable Basis & Non causal", "-"*10)
    enc_basis, dec_basis = 'trainable', 'trainable'
    enc_nonlinear = 'relu'

    causal = False
    mask_nonlinear = 'sigmoid'
    n_sources = 2

    model = GALRNet(
        D, kernel_size=M, enc_basis=enc_basis, dec_basis=dec_basis, enc_nonlinear=enc_nonlinear,
        sep_hidden_channels=H,
        sep_chunk_size=K, sep_hop_size=P, sep_down_chunk_size=Q,
        sep_num_blocks=N, sep_num_heads=J,
        sep_norm=sep_norm, mask_nonlinear=mask_nonlinear,
        low_dimension=low_dimension,
        causal=causal,
        n_sources=n_sources
    )
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

    model = GALRNet(
        D, kernel_size=M, enc_basis=enc_basis, dec_basis=dec_basis, window_fn=window_fn,
        sep_hidden_channels=H,
        sep_chunk_size=K, sep_hop_size=P, sep_down_chunk_size=Q,
        sep_num_blocks=N, sep_num_heads=J,
        sep_norm=sep_norm, mask_nonlinear=mask_nonlinear,
        causal=causal,
        n_sources=n_sources
    )
    print(model)
    print("# Parameters: {}".format(model.num_parameters))

    output = model(input)
    print(input.size(), output.size())

def _test_galrnet_paper():
    batch_size = 2
    K, P, Q = 100, 50, 32

    # Encoder & decoder
    C, T = 1, 1024
    M, D = 16, 64

    # Separator
    H = 128 # for each direction
    N = 6
    J = 8
    sep_norm = True

    low_dimension=True

    input = torch.randn((batch_size, C, T), dtype=torch.float)

    enc_basis, dec_basis = 'trainable', 'trainable'
    enc_nonlinear = None

    causal = False
    mask_nonlinear = 'relu'
    n_sources = 2

    model = GALRNet(
        D, kernel_size=M, enc_basis=enc_basis, dec_basis=dec_basis, enc_nonlinear=enc_nonlinear,
        sep_hidden_channels=H,
        sep_chunk_size=K, sep_hop_size=P, sep_down_chunk_size=Q,
        sep_num_blocks=N, sep_num_heads=J,
        sep_norm=sep_norm, mask_nonlinear=mask_nonlinear,
        low_dimension=low_dimension,
        n_sources=n_sources,
        causal=causal
    )
    print(model)
    print("# Parameters: {}".format(model.num_parameters))

    output = model(input)
    print(input.size(), output.size())

def _test_big_galrnet_paper():
    batch_size = 2
    K, P, Q = 100, 50, 32

    # Encoder & decoder
    C, T = 1, 1024
    M, D = 16, 128

    # Separator
    H = 128 # for each direction
    N = 6
    J = 8
    sep_norm = True

    low_dimension=True

    input = torch.randn((batch_size, C, T), dtype=torch.float)

    enc_basis, dec_basis = 'trainable', 'trainable'
    enc_nonlinear = None

    causal = False
    mask_nonlinear = 'relu'
    n_sources = 2

    model = GALRNet(
        D, kernel_size=M, enc_basis=enc_basis, dec_basis=dec_basis, enc_nonlinear=enc_nonlinear,
        sep_hidden_channels=H,
        sep_chunk_size=K, sep_hop_size=P, sep_down_chunk_size=Q,
        sep_num_blocks=N, sep_num_heads=J,
        sep_norm=sep_norm, mask_nonlinear=mask_nonlinear,
        low_dimension=low_dimension,
        n_sources=n_sources,
        causal=causal
    )
    print(model)
    print("# Parameters: {}".format(model.num_parameters))

    output = model(input)
    print(input.size(), output.size())

if __name__ == '__main__':
    print("="*10, "Separator", "="*10)
    _test_separator()
    print()

    print("="*10, "GALRNet", "="*10)
    _test_galrnet()
    print()

    print("="*10, "GALRNet (same configuration in paper)", "="*10)
    _test_galrnet_paper()
    print()

    print("="*10, "Bigger GALRNet (same configuration in paper)", "="*10)
    _test_big_galrnet_paper()
    print()