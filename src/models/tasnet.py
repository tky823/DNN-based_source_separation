import torch
import torch.nn as nn
import torch.nn.functional as F

from models.filterbank import FourierEncoder, FourierDecoder, Decoder, GatedEncoder

EPS = 1e-12

class TasNetBase(nn.Module):
    def __init__(self, kernel_size, stride=None, window_fn='hann', trainable_enc=False, trainable_dec=False):
        super().__init__()
        
        assert kernel_size % stride == 0, "kernel_size is expected divisible by stride"
        
        self.kernel_size, self.stride = kernel_size, stride
        
        self.encoder = FourierEncoder(kernel_size, stride=stride, window_fn=window_fn, trainable=trainable_enc)
        self.decoder = FourierDecoder(kernel_size, stride=stride, window_fn=window_fn, trainable=trainable_dec)
    
    def forward(self, input):
        """
        Args:
            input (batch_size, 1, T)
        Returns:
            output (batch_size, 1, T)
        """
        output, _ = self.extract_latent(input)
        
        return output
    
    def extract_latent(self, input):
        """
        Args:
            input (batch_size, 1, T)
        Returns:
            output (batch_size, 1, T)
        """
        _, C_in, T = input.size()
        
        assert C_in == 1, "input.size() is expected (?, 1, ?), but given {}".format(input.size())
        
        kernel_size, stride = self.kernel_size, self.stride
        
        padding = (stride - (T - kernel_size) % stride) % stride + 2 * kernel_size # Assumes that "kernel_size % stride is 0"
        padding_left = padding // 2
        padding_right = padding - padding_left
        
        input = F.pad(input, (padding_left, padding_right))
        latent = self.encoder(input)
        output = self.decoder(latent)
        output = F.pad(output, (-padding_left, -padding_right))
        
        return output, latent
    
    @property
    def num_parameters(self):
        _num_parameters = 0
        
        for p in self.parameters():
            if p.requires_grad:
                _num_parameters += p.numel()
                
        return _num_parameters

class TasNet(nn.Module):
    """
    LSTM-TasNet
    """
    def __init__(self, n_basis, kernel_size=16, stride=8, sep_num_blocks=2, sep_num_layers=2, sep_hidden_channels=1000,
        causal=False,
        n_sources=2,
        eps=EPS,
        **kwargs
    ):
        super().__init__()
        
        assert kernel_size % stride == 0, "kernel_size is expected divisible by stride"
        
        if 'in_channels' in kwargs:
            self.in_channels = kwargs['in_channels']
        else:
            self.in_channels = 1
        self.n_basis = n_basis
        self.kernel_size, self.stride = kernel_size, stride
        self.sep_num_blocks, self.sep_num_layers = sep_num_blocks, sep_num_layers
        self.causal = causal
        self.n_sources = n_sources
        self.eps = eps
        
        self.encoder = GatedEncoder(self.in_channels, n_basis, kernel_size=kernel_size, stride=stride, eps=eps)
        self.separator = Separator(n_basis, num_blocks=sep_num_blocks, num_layers=sep_num_layers, hidden_channels=sep_hidden_channels, causal=causal, n_sources=n_sources)
        self.decoder = Decoder(n_basis, self.in_channels, kernel_size=kernel_size, stride=stride)
    
    def forward(self, input):
        """
        Args:
            input (batch_size, 1, T)
        Returns:
            output (batch_size, n_sources, T)
        """
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
    
    @property
    def num_parameters(self):
        _num_parameters = 0
        
        for p in self.parameters():
            if p.requires_grad:
                _num_parameters += p.numel()
                
        return _num_parameters
    
    def get_package(self):
        package = {
            'in_channels': self.in_channels,
            'n_basis': self.n_basis,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'sep_num_blocks': self.sep_num_blocks,
            'sep_num_layers': self.sep_num_layers,
            'causal': self.causal,
            'n_sources': self.n_sources,
            'eps': self.eps
        }
        
        return package

"""
    Modules for LSTM-TasNet
"""

class Separator(nn.Module):
    """
    Default separator of TasNet.
    """
    def __init__(self, n_basis, num_blocks, num_layers, hidden_channels, causal=False, n_sources=2, eps=EPS):
        super().__init__()
        
        self.num_blocks = num_blocks
        self.n_basis, self.n_sources = n_basis, n_sources
        self.eps = eps
        
        hidden_channels = n_sources*n_basis
        
        self.gamma = nn.Parameter(torch.Tensor(1, n_basis, 1))
        self.beta = nn.Parameter(torch.Tensor(1, n_basis, 1))
        
        net = []
        
        for idx in range(num_blocks):
            if idx == 0:
                net.append(LSTMBlock(n_basis, n_sources*n_basis, hidden_channels=hidden_channels, num_layers=num_layers, causal=causal))
            else:
                net.append(LSTMBlock(n_sources*n_basis, n_sources*n_basis, hidden_channels=hidden_channels, num_layers=num_layers))
            
        self.net = nn.Sequential(*net)
            
        self._reset_parameters()

    def _reset_parameters(self):
        self.gamma.data.fill_(1)
        self.beta.data.zero_()
    
    def forward(self, input):
        num_blocks = self.num_blocks
        n_basis, n_sources = self.n_basis, self.n_sources
        eps = self.eps
        
        batch_size, _, n_frames = input.size()
    
        mean = input.mean(dim=1, keepdim=True)
        squared_mean = (input**2).mean(dim=1, keepdim=True)
        var = squared_mean - mean**2
        x = self.gamma * (input - mean)/(torch.sqrt(var) + eps) + self.beta
        
        skip = 0
        
        for idx in range(num_blocks):
            x = self.net[idx](x)
            skip = x + skip
            
        x = skip
        output = x.view(batch_size, n_sources, n_basis, n_frames)

        return output

class LSTMBlock(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=None, num_layers=2, causal=False):
        super().__init__()
        
        if hidden_channels is None:
            hidden_channels = out_channels
        
        self.num_layers = num_layers
        
        net = []
        
        for idx in range(num_layers):
            if idx == 0:
                net.append(LSTMLayer(in_channels, hidden_channels, causal=causal))
            elif idx == num_layers - 1:
                net.append(LSTMLayer(hidden_channels, out_channels, causal=causal))
            else:
                net.append(LSTMLayer(hidden_channels, hidden_channels, causal=causal))
            
        self.net = nn.Sequential(*net)
    
    def forward(self, input):
        """
        Args:
            input (batch_size, in_channels, T)
        Returns:
            output (batch_size, out_channels, T)
        """
        output = self.net(input)
        
        return output

class LSTMLayer(nn.Module):
    def __init__(self, in_channels, out_channels, causal=False):
        super().__init__()
        
        if causal:
            num_directions = 1
            bidirectional = False
        else:
            num_directions = 2
            bidirectional = True
        
        self.rnn = nn.LSTM(in_channels, out_channels//num_directions, batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Linear(out_channels, out_channels)
    
    def forward(self, input):
        """
        Args:
            input (batch_size, in_channels, T)
        Returns:
            output (batch_size, out_channels, T)
        """
        x = input.permute(0, 2, 1) # -> (batch_size, T, in_channels)
        x, (_, _) = self.rnn(x) # -> (batch_size, T, out_channels//num_directions)
        x = self.fc(x) # -> (batch_size, T, out_channels)
        output = x.permute(0, 2, 1) # -> (batch_size, out_channels, T)
        
        return output

def _test_tasnet_base():
    torch.manual_seed(111)
    
    batch_size = 2
    C = 1
    T = 64
    kernel_size, stride = 8, 2
    
    input = torch.randn((batch_size, C, T), dtype=torch.float)
    
    window_fn = 'hamming'
    
    model = TasNetBase(kernel_size=kernel_size, stride=stride, window_fn=window_fn)
    output = model(input)
    print(input.size(), output.size())

    plt.figure()
    plt.plot(range(T), input[0, 0].detach().numpy())
    plt.plot(range(T), output[0, 0].detach().numpy())
    plt.savefig('data/tasnet/Fourier.png', bbox_inches='tight')
    plt.close()
    
    basis = model.decoder.get_basis()
    print(basis.size())
    
    plt.figure()
    plt.pcolormesh(basis.detach().cpu().numpy(), cmap='bwr', norm=Normalize(vmin=-1, vmax=1))
    plt.colorbar()
    plt.savefig('data/tasnet/basis.png', bbox_inches='tight')
    plt.close()

    _, latent = model.extract_latent(input)
    print(latent.size())
    power = torch.abs(latent)
    
    plt.figure()
    plt.pcolormesh(power[0].detach().cpu().numpy(), cmap='bwr')
    plt.colorbar()
    plt.savefig('data/tasnet/power.png', bbox_inches='tight')
    plt.close()

def _test_tasnet():
    batch_size = 2
    C = 1
    T = 64
    kernel_size, stride = 8, 2
    repeat = 2
    n_basis = kernel_size * repeat * 2
    
    input = torch.randn((batch_size, C, T), dtype=torch.float)

    # LSTM-TasNet configuration
    sep_num_blocks, sep_num_layers, sep_hidden_channels = 2, 2, 32
    n_sources = 3
    
    # Non causal
    print("-"*10, "Non causal", "-"*10)
    causal = False

    model = TasNet(n_basis, kernel_size=kernel_size, stride=stride, sep_num_blocks=sep_num_blocks, sep_num_layers=sep_num_layers, sep_hidden_channels=sep_hidden_channels, causal=causal, n_sources=n_sources)
    print(model)
    print("# Parameters: {}".format(model.num_parameters))
    
    output = model(input)
    print(input.size(), output.size())
    print()
    
    # Causal
    print("-"*10, "Causal", "-"*10)
    causal = True
    
    model = TasNet(n_basis, kernel_size=kernel_size, stride=stride, sep_num_blocks=sep_num_blocks, sep_num_layers=sep_num_layers, sep_hidden_channels=sep_hidden_channels, causal=causal, n_sources=n_sources)
    print(model)
    print("# Parameters: {}".format(model.num_parameters))

    output = model(input)
    print(input.size(), output.size())

def _test_multichannel_tasnet():
    batch_size = 2
    C = 2
    T = 64
    kernel_size, stride = 8, 2
    repeat = 2
    n_basis = kernel_size * repeat * 2
    
    input = torch.randn((batch_size, 1, C, T), dtype=torch.float)

    # LSTM-TasNet configuration
    sep_num_blocks, sep_num_layers, sep_hidden_channels = 2, 2, 32
    n_sources = 3
    
    # Non causal
    print("-"*10, "Non causal", "-"*10)
    causal = False

    model = TasNet(n_basis, in_channels=C, kernel_size=kernel_size, stride=stride, sep_num_blocks=sep_num_blocks, sep_num_layers=sep_num_layers, sep_hidden_channels=sep_hidden_channels, causal=causal, n_sources=n_sources)
    print(model)
    print("# Parameters: {}".format(model.num_parameters))
    
    output = model(input)
    print(input.size(), output.size())

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    
    print("="*10, "TasNet-Base", "="*10)
    _test_tasnet_base()
    print()
    
    print("="*10, "LSTM-TasNet", "="*10)
    _test_tasnet()
    print()
    
    print("="*10, "LSTM-TasNet (multichannel)", "="*10)
    _test_multichannel_tasnet()
