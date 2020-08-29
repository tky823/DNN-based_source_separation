import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils_tasnet import choose_basis, choose_layer_norm
from models.tcn import TemporalConvNet

EPS=1e-12

class ConvTasNet(nn.Module):
    def __init__(self, n_basis, kernel_size, stride=None, enc_basis=None, dec_basis=None, sep_hidden_channels=256, sep_bottleneck_channels=128, sep_skip_channels=128, sep_kernel_size=3, sep_num_blocks=3, sep_num_layers=8, dilated=True, separable=True, causal=True, sep_nonlinear='prelu', sep_norm=True, mask_nonlinear='sigmoid', n_sources=2, eps=EPS, **kwargs):
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
        self.sep_hidden_channels, self.sep_bottleneck_channels, self.sep_skip_channels = sep_hidden_channels, sep_bottleneck_channels, sep_skip_channels
        self.sep_kernel_size = sep_kernel_size
        self.sep_num_blocks, self.sep_num_layers = sep_num_blocks, sep_num_layers
        
        self.dilated, self.separable, self.causal = dilated, separable, causal
        self.sep_nonlinear, self.sep_norm = sep_nonlinear, sep_norm
        self.mask_nonlinear = mask_nonlinear
        
        self.n_sources = n_sources
        self.eps = eps
        
        # Network configuration
        encoder, decoder = choose_basis(n_basis, kernel_size=kernel_size, stride=stride, enc_basis=enc_basis, dec_basis=dec_basis, **kwargs)
        
        self.encoder = encoder
        self.separator = Separator(n_basis, bottleneck_channels=sep_bottleneck_channels, hidden_channels=sep_hidden_channels, skip_channels=sep_skip_channels, kernel_size=sep_kernel_size, num_blocks=sep_num_blocks, num_layers=sep_num_layers, dilated=dilated, separable=separable, causal=causal, nonlinear=sep_nonlinear, norm=sep_norm, mask_nonlinear=mask_nonlinear, n_sources=n_sources, eps=eps)
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
        
        return package
    
    @classmethod
    def build_model(cls, model_path):
        package = torch.load(model_path, map_location=lambda storage, loc: storage)
        
        n_basis = package['n_basis']
        kernel_size, stride = package['kernel_size'], package['stride']
        enc_basis, dec_basis = package['enc_basis'], package['dec_basis']
        enc_nonlinear = package['enc_nonlinear']
        window_fn = package['window_fn']
        
        sep_hidden_channels, sep_bottleneck_channels, sep_skip_channels = package['sep_hidden_channels'], package['sep_bottleneck_channels'], package['sep_skip_channels']
        sep_kernel_size = package['sep_kernel_size']
        sep_num_blocks, sep_num_layers = package['sep_num_blocks'], package['sep_num_layers']
        
        dilated, separable, causal = package['dilated'], package['separable'], package['causal']
        sep_nonlinear, sep_norm = package['sep_nonlinear'], package['sep_norm']
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
    def __init__(self, num_features, bottleneck_channels=128, hidden_channels=256, skip_channels=128, kernel_size=3, num_blocks=3, num_layers=8, dilated=True, separable=True, causal=True, nonlinear='prelu', norm=True, mask_nonlinear='sigmoid', n_sources=2, eps=EPS):
        super().__init__()
        
        self.num_features, self.n_sources = num_features, n_sources
        
        self.norm1d = choose_layer_norm(num_features, causal=causal, eps=eps)
        self.bottleneck_conv1d = nn.Conv1d(num_features, bottleneck_channels, kernel_size=1, stride=1)
        self.tcn = TemporalConvNet(bottleneck_channels, hidden_channels=hidden_channels, skip_channels=skip_channels, kernel_size=kernel_size, num_blocks=num_blocks, num_layers=num_layers, dilated=dilated, separable=separable, causal=causal, nonlinear=nonlinear, norm=norm)
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
            input (batch_size, num_features, T_bin)
        Returns:
            output (batch_size, n_sources, n_basis, T_bin)
        """
        num_features, n_sources = self.num_features, self.n_sources

        batch_size, _, T_bin = input.size()

        x = self.norm1d(input)
        x = self.bottleneck_conv1d(x)
        x = self.tcn(x)
        x = self.prelu(x)
        x = self.mask_conv1d(x)
        x = self.mask_nonlinear(x)
        output = x.view(batch_size, n_sources, num_features, T_bin)
        
        return output
        
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    
    torch.manual_seed(111)
    
    batch_size = 4
    C = 1
    T = 64
    
    input = torch.randn((batch_size, C, T), dtype=torch.float)
    
    print("="*10, "Conv-TasNet", "="*10)
    # Conv-TasNet configuration
    L, stride = 8, 2
    
    N = 64
    H, B, Sc = 128, 64, 64
    P = 3
    R, X = 4, 3
    sep_norm = True
    
    # Non causal
    print("-"*10, "Trainable Basis & Non causal", "-"*10)
    enc_basis, dec_basis = 'trainable', 'trainable'
    enc_nonlinear = 'relu'
    causal = False
    mask_nonlinear = 'sigmoid'
    n_sources = 2

    model = ConvTasNet(N, kernel_size=L, stride=stride, enc_basis=enc_basis, dec_basis=dec_basis, enc_nonlinear=enc_nonlinear, sep_hidden_channels=H, sep_bottleneck_channels=B, sep_skip_channels=Sc, sep_kernel_size=P, sep_num_blocks=R, sep_num_layers=X, causal=causal, sep_norm=sep_norm, mask_nonlinear=mask_nonlinear, n_sources=n_sources)
    print(model)
    print("# Parameters: {}".format(model.num_parameters))
    
    output = model(input)
    print(input.size(), output.size())
    print()
    
    basis = model.encoder.get_basis()
    
    plt.figure()
    plt.pcolormesh(basis, cmap='bwr', norm=Normalize(vmin=-1, vmax=1))
    plt.colorbar()
    plt.savefig('data/basis_enc-trainable.png')
    plt.close()
    
    # Causal
    print("-"*10, "Fourier Basis & Causal", "-"*10)
    enc_basis, dec_basis = 'Fourier', 'Fourier'
    causal = True
    mask_nonlinear = 'softmax'
    window_fn = 'hamming'
    n_sources = 3

    model = ConvTasNet(N, kernel_size=L, stride=stride, enc_basis=enc_basis, dec_basis=dec_basis, window_fn=window_fn, sep_hidden_channels=H, sep_bottleneck_channels=B, sep_skip_channels=Sc, sep_kernel_size=P, sep_num_blocks=R, sep_num_layers=X, causal=causal, sep_norm=sep_norm, mask_nonlinear=mask_nonlinear, n_sources=n_sources)
    print(model)
    print("# Parameters: {}".format(model.num_parameters))
    
    output = model(input)
    print(input.size(), output.size())
    
    basis = model.encoder.get_basis()
    
    plt.figure()
    plt.pcolormesh(basis, cmap='bwr', norm=Normalize(vmin=-1, vmax=1))
    plt.colorbar()
    plt.savefig('data/basis_enc-Fourier.png')
    plt.close()
