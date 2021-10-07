import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils_filterbank import choose_filterbank
from utils.utils_tasnet import choose_layer_norm
from models.tcn import TemporalConvNet

EPS = 1e-12

class ConvTasNet(nn.Module):
    def __init__(self,
        n_basis, kernel_size, stride=None, enc_basis=None, dec_basis=None,
        sep_hidden_channels=256, sep_bottleneck_channels=128, sep_skip_channels=128, sep_kernel_size=3, sep_num_blocks=3, sep_num_layers=8,
        dilated=True, separable=True,
        sep_nonlinear='prelu', sep_norm=True, mask_nonlinear='sigmoid',
        causal=True,
        n_sources=2,
        eps=EPS,
        **kwargs
    ):
        super().__init__()
        
        if stride is None:
            stride = kernel_size // 2
        
        assert kernel_size % stride == 0, "kernel_size is expected divisible by stride"
        
        # Encoder-decoder
        if 'in_channels' in kwargs:
            self.in_channels = kwargs['in_channels']
        else:
            self.in_channels = 1
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
        self.sep_hidden_channels, self.sep_bottleneck_channels, self.sep_skip_channels = sep_hidden_channels, sep_bottleneck_channels, sep_skip_channels
        self.sep_kernel_size = sep_kernel_size
        self.sep_num_blocks, self.sep_num_layers = sep_num_blocks, sep_num_layers
        
        self.dilated, self.separable, self.causal = dilated, separable, causal
        self.sep_nonlinear, self.sep_norm = sep_nonlinear, sep_norm
        self.mask_nonlinear = mask_nonlinear
        
        self.n_sources = n_sources
        self.eps = eps
        
        # Network configuration
        encoder, decoder = choose_filterbank(n_basis, kernel_size=kernel_size, stride=stride, enc_basis=enc_basis, dec_basis=dec_basis, **kwargs)
        
        self.encoder = encoder
        self.separator = Separator(
            n_basis, bottleneck_channels=sep_bottleneck_channels, hidden_channels=sep_hidden_channels, skip_channels=sep_skip_channels,
            kernel_size=sep_kernel_size, num_blocks=sep_num_blocks, num_layers=sep_num_layers,
            dilated=dilated, separable=separable, causal=causal, nonlinear=sep_nonlinear, norm=sep_norm, mask_nonlinear=mask_nonlinear,
            n_sources=n_sources, eps=eps
        )
        self.decoder = decoder
    
    def forward(self, input):
        output, _ = self.extract_latent(input)
        
        return output
    
    def extract_latent(self, input):
        """
        Args:
            input (batch_size, C_in, T)
        Returns:
            output (batch_size, n_sources, T) or (batch_size, n_sources, C_in, T)
            latent (batch_size, n_sources, n_basis, T'), where T' = (T-K)//S+1
        """
        n_sources = self.n_sources
        n_basis = self.n_basis
        kernel_size, stride = self.kernel_size, self.stride
        
        n_dims = input.dim()

        if n_dims == 3:
            batch_size, C_in, T = input.size()
            assert C_in == 1, "input.size() is expected (?, 1, ?), but given {}".format(input.size())
        elif n_dims == 4:
            batch_size, C_in, n_mics, T = input.size()
            assert C_in == 1, "input.size() is expected (?, 1, ?, ?), but given {}".format(input.size())
            input = input.view(batch_size, n_mics, T)
        else:
            raise ValueError("Not support {} dimension input".format(n_dims))
        
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
        if n_dims == 3:
            x_hat = x_hat.view(batch_size, n_sources, -1)
        else: # n_dims == 4
            x_hat = x_hat.view(batch_size, n_sources, n_mics, -1)
        output = F.pad(x_hat, (-padding_left, -padding_right))
        
        return output, latent
    
    def get_config(self):
        config = {
            'in_channels': self.in_channels,
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
            'sep_skip_channels': self.sep_skip_channels,
            'sep_kernel_size': self.sep_kernel_size,
            'sep_num_blocks': self.sep_num_blocks,
            'sep_num_layers': self.sep_num_layers,
            'dilated': self.dilated,
            'separable': self.separable,
            'causal': self.causal,
            'sep_nonlinear': self.sep_nonlinear,
            'sep_norm': self.sep_norm,
            'mask_nonlinear': self.mask_nonlinear,
            'n_sources': self.n_sources,
            'eps': self.eps
        }
        
        return config

    def get_package(self):
        return self.get_config()
    
    @classmethod
    def build_model(cls, model_path, load_state_dict=False):
        config = torch.load(model_path, map_location=lambda storage, loc: storage)
        
        in_channels = config.get('in_channels') or 1
        n_basis = config.get('n_bases') or config['n_basis']
        kernel_size, stride = config['kernel_size'], config['stride']
        enc_basis, dec_basis = config.get('enc_bases') or config['enc_basis'], config.get('dec_bases') or config['dec_basis']
        enc_nonlinear = config['enc_nonlinear']
        enc_onesided, enc_return_complex = config.get('enc_onesided') or None, config.get('enc_return_complex') or None
        window_fn = config['window_fn']
        
        sep_hidden_channels, sep_bottleneck_channels, sep_skip_channels = config['sep_hidden_channels'], config['sep_bottleneck_channels'], config['sep_skip_channels']
        sep_kernel_size = config['sep_kernel_size']
        sep_num_blocks, sep_num_layers = config['sep_num_blocks'], config['sep_num_layers']
        
        dilated, separable, causal = config['dilated'], config['separable'], config['causal']
        sep_nonlinear, sep_norm = config['sep_nonlinear'], config['sep_norm']
        mask_nonlinear = config['mask_nonlinear']
        
        n_sources = config['n_sources']
        
        eps = config['eps']
        
        model = cls(
            n_basis, in_channels=in_channels, kernel_size=kernel_size, stride=stride, enc_basis=enc_basis, dec_basis=dec_basis, enc_nonlinear=enc_nonlinear,
            window_fn=window_fn, enc_onesided=enc_onesided, enc_return_complex=enc_return_complex,
            sep_hidden_channels=sep_hidden_channels, sep_bottleneck_channels=sep_bottleneck_channels, sep_skip_channels=sep_skip_channels,
            sep_kernel_size=sep_kernel_size, sep_num_blocks=sep_num_blocks, sep_num_layers=sep_num_layers,
            dilated=dilated, separable=separable, causal=causal, sep_nonlinear=sep_nonlinear, sep_norm=sep_norm, mask_nonlinear=mask_nonlinear,
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
        self, num_features, bottleneck_channels=128, hidden_channels=256, skip_channels=128, kernel_size=3, num_blocks=3, num_layers=8,
        dilated=True, separable=True, causal=True, nonlinear='prelu', norm=True, mask_nonlinear='sigmoid',
        n_sources=2,
        eps=EPS
    ):
        super().__init__()
        
        self.num_features, self.n_sources = num_features, n_sources
        
        norm_name = 'cLN' if causal else 'gLN'
        self.norm1d = choose_layer_norm(norm_name, num_features, causal=causal, eps=eps)
        self.bottleneck_conv1d = nn.Conv1d(num_features, bottleneck_channels, kernel_size=1, stride=1)
        self.tcn = TemporalConvNet(
            bottleneck_channels, hidden_channels=hidden_channels, skip_channels=skip_channels, kernel_size=kernel_size, num_blocks=num_blocks, num_layers=num_layers,
            dilated=dilated, separable=separable, causal=causal, nonlinear=nonlinear, norm=norm
        )
        self.prelu = nn.PReLU()
        self.mask_conv1d = nn.Conv1d(skip_channels, n_sources*num_features, kernel_size=1, stride=1)
        
        if mask_nonlinear == 'sigmoid':
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
            output (batch_size, n_sources, n_basis, n_frames)
        """
        num_features, n_sources = self.num_features, self.n_sources

        batch_size, _, n_frames = input.size()

        x = self.norm1d(input)
        x = self.bottleneck_conv1d(x)
        x = self.tcn(x)
        x = self.prelu(x)
        x = self.mask_conv1d(x)
        x = self.mask_nonlinear(x)
        output = x.view(batch_size, n_sources, num_features, n_frames)
        
        return output

def _test_conv_tasnet():
    batch_size = 4
    C = 1
    T = 64
    
    input = torch.randn((batch_size, C, T), dtype=torch.float)

    H, B, Sc = 128, 64, 64
    P = 3
    R, X = 3, 8
    sep_norm = True

    # Causal
    print("-"*10, "Fourier Basis & Causal", "-"*10)
    L, stride = 16, 8
    N = L // 2 + 1
    enc_basis, dec_basis = 'Fourier', 'Fourier'
    causal = True
    mask_nonlinear = 'softmax'
    window_fn = 'hamming'
    onesided, return_complex = True, True
    n_sources = 3

    model = ConvTasNet(
        N, kernel_size=L, stride=stride, enc_basis=enc_basis, dec_basis=dec_basis,
        window_fn=window_fn, onesided=onesided, return_complex=return_complex,
        sep_hidden_channels=H, sep_bottleneck_channels=B, sep_skip_channels=Sc,
        sep_kernel_size=P, sep_num_blocks=R, sep_num_layers=X,
        causal=causal, sep_norm=sep_norm, mask_nonlinear=mask_nonlinear,
        n_sources=n_sources
    )
    print(model)
    print("# Parameters: {}".format(model.num_parameters))
    
    output = model(input)
    print(input.size(), output.size())
    print()
    
    basis = model.encoder.get_basis()
    
    plt.figure()
    plt.pcolormesh(basis.detach().cpu().numpy(), cmap='bwr', norm=Normalize(vmin=-1, vmax=1))
    plt.colorbar()
    plt.savefig('data/conv-tasnet/basis_enc-Fourier.png', bbox_inches='tight')
    plt.close()

    # Non causal
    print("-"*10, "Fourier Basis & Non-causal", "-"*10)
    L, stride = 16, 4
    N = 2 * L
    enc_basis, dec_basis = 'Fourier', 'Fourier'
    causal = False
    mask_nonlinear = 'softmax'
    window_fn = 'hann'
    onesided, return_complex = False, False
    n_sources = 2

    model = ConvTasNet(
        N, kernel_size=L, stride=stride, enc_basis=enc_basis, dec_basis=dec_basis,
        window_fn=window_fn, onesided=onesided, return_complex=return_complex,
        sep_hidden_channels=H, sep_bottleneck_channels=B, sep_skip_channels=Sc,
        sep_kernel_size=P, sep_num_blocks=R, sep_num_layers=X,
        causal=causal, sep_norm=sep_norm, mask_nonlinear=mask_nonlinear,
        n_sources=n_sources
    )
    print(model)
    print("# Parameters: {}".format(model.num_parameters))
    
    output = model(input)
    print(input.size(), output.size())
    print()

    # Pseudo inverse
    print("-"*10, "Decoder is a pseudo inverse of encoder", "-"*10)
    L, stride = 16, 4
    N = 64
    enc_basis, dec_basis = 'trainable', 'pinv'
    enc_nonlinear = None
    causal = False
    mask_nonlinear = 'sigmoid'
    n_sources = 2

    model = ConvTasNet(
        N, kernel_size=L, stride=stride, enc_basis=enc_basis, dec_basis=dec_basis, enc_nonlinear=enc_nonlinear,
        sep_hidden_channels=H, sep_bottleneck_channels=B, sep_skip_channels=Sc,
        sep_kernel_size=P, sep_num_blocks=R, sep_num_layers=X,
        causal=causal, sep_norm=sep_norm, mask_nonlinear=mask_nonlinear, n_sources=n_sources
    )
    print(model)
    print("# Parameters: {}".format(model.num_parameters))
    
    output = model(input)
    print(input.size(), output.size())
    
    weight = model.encoder.get_basis().detach().cpu().numpy()
    weight_pinverse = model.decoder.get_basis().detach().cpu().numpy().T
    reconstruction = np.matmul(weight_pinverse, weight)
    print(reconstruction)

def _test_multichannel_conv_tasnet():
    batch_size = 4
    C = 2
    T = 64
    
    input = torch.randn((batch_size, 1, C, T), dtype=torch.float)

    L, stride = 16, 8
    N = 64
    H, B, Sc = 128, 64, 64
    P = 3
    R, X = 3, 8
    sep_norm = True

    enc_basis, dec_basis = 'trainable', 'trainable'
    enc_nonlinear = 'relu'
    causal = False
    mask_nonlinear = 'softmax'
    n_sources = 3

    model = ConvTasNet(
        N, in_channels=C, kernel_size=L, stride=stride, enc_basis=enc_basis, dec_basis=dec_basis, enc_nonlinear=enc_nonlinear,
        sep_hidden_channels=H, sep_bottleneck_channels=B, sep_skip_channels=Sc,
        sep_kernel_size=P, sep_num_blocks=R, sep_num_layers=X, causal=causal, sep_norm=sep_norm, mask_nonlinear=mask_nonlinear,
        n_sources=n_sources
    )
    print(model)
    print("# Parameters: {}".format(model.num_parameters))
    
    output = model(input)
    print(input.size(), output.size())
    print()

def _test_conv_tasnet_paper():
    batch_size = 4
    C = 1
    T = 64
    
    input = torch.randn((batch_size, C, T), dtype=torch.float)

    L, stride = 16, 8
    N = 512
    H, B, Sc = 512, 128, 128
    P = 3
    R, X = 3, 8
    sep_norm = True
    
    # Non causal
    print("-"*10, "Trainable Basis & Non causal", "-"*10)
    enc_basis, dec_basis = 'trainable', 'trainable'
    enc_nonlinear = None
    causal = False
    mask_nonlinear = 'sigmoid'
    n_sources = 2

    model = ConvTasNet(
        N, kernel_size=L, stride=stride, enc_basis=enc_basis, dec_basis=dec_basis, enc_nonlinear=enc_nonlinear,
        sep_hidden_channels=H, sep_bottleneck_channels=B, sep_skip_channels=Sc, sep_kernel_size=P, sep_num_blocks=R, sep_num_layers=X,
        causal=causal, sep_norm=sep_norm, mask_nonlinear=mask_nonlinear,
        n_sources=n_sources
    )
    print(model)
    print("# Parameters: {}".format(model.num_parameters))
    
    output = model(input)
    print(input.size(), output.size())

    basis = model.encoder.get_basis()

    plt.figure()
    plt.pcolormesh(basis.detach().cpu().numpy(), cmap='bwr', norm=Normalize(vmin=-1, vmax=1))
    plt.colorbar()
    plt.savefig('data/conv-tasnet/basis_enc-trainable.png', bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    
    torch.manual_seed(111)

    print("="*10, "Conv-TasNet", "="*10)
    _test_conv_tasnet()
    print()

    print("="*10, "Conv-TasNet (multichannel)", "="*10)
    _test_multichannel_conv_tasnet()
    print()

    print("="*10, "Conv-TasNet (same configuration in the paper)", "="*10)
    _test_conv_tasnet_paper()
    print()