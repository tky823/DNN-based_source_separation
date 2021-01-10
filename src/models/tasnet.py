import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils_audio import build_Fourier_bases, build_window, build_optimal_window

EPS=1e-12

class TasNetBase(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size, stride=None, window_fn='hann', trainable_enc=False, trainable_dec=False):
        super().__init__()
        
        assert kernel_size%stride == 0, "kernel_size is expected divisible by stride"
        
        self.kernel_size, self.stride = kernel_size, stride
        
        self.encoder = FourierEncoder(in_channels, hidden_channels, kernel_size, stride=stride, window_fn=window_fn, trainable=trainable_enc)
        self.decoder = FourierDecoder(hidden_channels, in_channels, kernel_size, stride=stride, window_fn=window_fn, trainable=trainable_dec)
        
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
        batch_size, C_in, T = input.size()
        
        assert C_in == 1, "input.size() is expected (?,1,?), but given {}".format(input.size())
        
        kernel_size, stride = self.kernel_size, self.stride
        
        padding = (stride - (T - kernel_size)%stride)%stride + 2 * kernel_size # Assume that "kernel_size%stride is 0"
        padding_left = padding // 2
        padding_right = padding - padding_left
        
        input = F.pad(input, (padding_left, padding_right))
        latent = self.encoder(input)
        output = self.decoder(latent)
        output = F.pad(output, (-padding_left, -padding_right))
        
        return output, latent
        
    def _get_num_parameters(self):
        num_parameters = 0
        
        for p in self.parameters():
            if p.requires_grad:
                num_parameters += p.numel()
                
        return num_parameters
        
class TasNet(nn.Module):
    """
    LSTM-TasNet
    """
    def __init__(self, in_channels, n_bases, kernel_size=16, stride=8, sep_num_blocks=2, sep_num_layers=2, sep_hidden_channels=1000, causal=False, n_sources=2, eps=EPS):
        super().__init__()
        
        assert kernel_size%stride == 0, "kernel_size is expected divisible by stride"
        
        self.in_channels = in_channels
        self.n_bases = n_bases
        self.kernel_size, self.stride = kernel_size, stride
        self.sep_num_blocks, self.sep_num_layers = sep_num_blocks, sep_num_layers
        self.causal = causal
        self.n_sources = n_sources
        self.eps = eps
        
        self.encoder = GatedEncoder(in_channels, n_bases, kernel_size=kernel_size, stride=stride, eps=eps)
        self.separator = Separator(n_bases, num_blocks=sep_num_blocks, num_layers=sep_num_layers, hidden_channels=sep_hidden_channels, causal=causal, n_sources=n_sources)
        self.decoder = Decoder(n_bases, in_channels, kernel_size=kernel_size, stride=stride)
        
        self.num_parameters = self._get_num_parameters()
        
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
            latent (batch_size, n_sources, n_bases, T'), where T' = (T-K)//S+1
        """
        n_sources = self.n_sources
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
        
    def get_package(self):
        package = {
            'in_channels': self.in_channels,
            'n_bases': self.n_bases,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'sep_num_blocks': self.sep_num_blocks,
            'sep_num_layers': self.sep_num_layers,
            'causal': self.causal,
            'n_sources': self.n_sources,
            'eps': self.eps
        }
        
        return package
        
        
class FourierEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=None, window_fn='hann', trainable=False):
        super().__init__()
         
        assert in_channels == 1, "in_channels is expected 1, given {}".format(in_channels)
        assert out_channels % (kernel_size*2) == 0, "out_channels % (kernel_size*2) is given {}".format(out_channels % (kernel_size*2))
        
        self.kernel_size, self.stride = kernel_size, stride
        
        repeat = out_channels//(kernel_size*2)
        self.repeat = repeat
    
        window = build_window(kernel_size, window_fn=window_fn) # (kernel_size,)

        cos_bases, sin_bases = build_Fourier_bases(kernel_size, normalize=True)
        cos_bases, sin_bases = cos_bases * window, - sin_bases * window
        
        bases = None
        
        for idx in range(repeat):
            rolled_cos_bases = torch.roll(cos_bases, kernel_size//repeat*idx, dims=1)
            rolled_sin_bases = torch.roll(sin_bases, kernel_size//repeat*idx, dims=1)
            if bases is None:
                bases = torch.cat([rolled_cos_bases, rolled_sin_bases], dim=0)
            else:
                bases = torch.cat([bases, rolled_cos_bases, rolled_sin_bases], dim=0)
        
        self.bases = nn.Parameter(bases.unsqueeze(dim=1), requires_grad=trainable)
        
    def forward(self, input):
        output = F.conv1d(input, self.bases, stride=self.stride)
        
        return output
    
    def extra_repr(self):
        s = "kernel_size={kernel_size}, stride={stride}"
        
        if self.repeat != 1:
            s += ", repeat={repeat}"
        
        return s.format(**self.__dict__)
        
    def get_bases(self):
        return self.bases.squeeze(dim=1).detach().cpu().numpy()

class FourierDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=None, window_fn='hann', trainable=False):
        super().__init__()
        
        assert out_channels == 1, "out_channels is expected 1, given {}".format(out_channels)
        assert in_channels % (kernel_size*2) == 0, "in_channels % (kernel_size*2) is given {}".format(in_channels % (kernel_size*2))
        
        self.kernel_size, self.stride = kernel_size, stride
        
        repeat = in_channels//(kernel_size*2)
        self.repeat = repeat
        
        window = build_window(kernel_size, window_fn=window_fn) # (kernel_size,)
        optimal_window = build_optimal_window(window, hop_size=stride)

        cos_bases, sin_bases = build_Fourier_bases(kernel_size, normalize=True)
        cos_bases, sin_bases = cos_bases * optimal_window / repeat, - sin_bases * optimal_window / repeat
        
        bases = None
        
        for idx in range(repeat):
            rolled_cos_bases = torch.roll(cos_bases, kernel_size//repeat*idx, dims=1)
            rolled_sin_bases = torch.roll(sin_bases, kernel_size//repeat*idx, dims=1)
            if bases is None:
                bases = torch.cat([rolled_cos_bases, rolled_sin_bases], dim=0)
            else:
                bases = torch.cat([bases, rolled_cos_bases, rolled_sin_bases], dim=0)
        
        self.bases = nn.Parameter(bases.unsqueeze(dim=1), requires_grad=trainable)
        
    def forward(self, input):
        output = F.conv_transpose1d(input, self.bases, stride=self.stride)
        
        return output
    
    def extra_repr(self):
        s = "kernel_size={kernel_size}, stride={stride}"
        
        if self.repeat != 1:
            s += ", repeat={repeat}"
        
        return s.format(**self.__dict__)
        
    def get_bases(self):
        return self.bases.squeeze(dim=1).detach().cpu().numpy()

class Encoder(nn.Module):
    def __init__(self, in_channels, n_bases, kernel_size=16, stride=8, nonlinear=None):
        super().__init__()
        
        assert in_channels == 1, "in_channels is expected 1, given {}".format(in_channels)
        
        self.kernel_size, self.stride = kernel_size, stride
        self.nonlinear = nonlinear
        
        self.conv1d = nn.Conv1d(in_channels, n_bases, kernel_size=kernel_size, stride=stride, bias=False)
        if nonlinear is not None:
            if nonlinear == 'relu':
                self.nonlinear1d = nn.ReLU()
            else:
                raise NotImplementedError("Not support {}".format(nonlinear))
            self.nonlinear = True
        else:
            self.nonlinear = False
        
        
    def forward(self, input):
        x = self.conv1d(input)
        
        if self.nonlinear:
            output = self.nonlinear1d(x)
        else:
            output = x
        
        return output
        
    def get_bases(self):
        bases = self.conv1d.weight.squeeze(dim=1).detach().cpu().numpy()
    
        return bases

class Decoder(nn.Module):
    def __init__(self, n_bases, out_channels, kernel_size=16, stride=8):
        super().__init__()
        
        assert out_channels == 1, "out_channels is expected 1, given {}".format(out_channels)
        
        self.kernel_size, self.stride = kernel_size, stride
        
        self.conv_transpose1d = nn.ConvTranspose1d(n_bases, out_channels, kernel_size=kernel_size, stride=stride, bias=False)
        
    def forward(self, input):
        output = self.conv_transpose1d(input)
        
        return output
        
    def get_bases(self):
        bases = self.conv_transpose1d.weight.squeeze(dim=1).detach().cpu().numpy()
        
        return bases

class PinvEncoder(nn.Module):
    def __init__(self, encoder: Encoder):
        super().__init__()

        if encoder.nonlinear:
            raise ValueError("Not support pseudo inverse of 'Conv1d + nonlinear'.")

        self.kernel_size, self.stride = encoder.kernel_size, encoder.stride
        self.weight = encoder.conv1d.weight

    def forward(self, input):
        kernel_size, stride = self.kernel_size, self.stride
        duplicate = kernel_size//stride
        weight = self.weight.permute(1,0,2).contiguous()
        weight_pinverse = torch.pinverse(weight).permute(2,0,1).contiguous() / duplicate

        output = F.conv_transpose1d(input, weight_pinverse, stride=stride)

        return output
    
    def get_bases(self):
        kernel_size, stride = self.kernel_size, self.stride
        duplicate = kernel_size//stride
        weight = self.weight.permute(1,0,2).contiguous()
        weight_pinverse = torch.pinverse(weight).permute(2,0,1).contiguous() / duplicate

        bases = weight_pinverse.squeeze(dim=1).detach().cpu().numpy()

        return bases

"""
    Modules for LSTM-TasNet
"""

class GatedEncoder(nn.Module):
    def __init__(self, in_channels, n_bases, kernel_size=16, stride=8, eps=EPS):
        super().__init__()
        
        self.kernel_size, self.stride = kernel_size, stride
        self.eps = eps
        
        self.conv1d_U = nn.Conv1d(in_channels, n_bases, kernel_size=kernel_size, stride=stride, bias=False)
        self.conv1d_V = nn.Conv1d(in_channels, n_bases, kernel_size=kernel_size, stride=stride, bias=False)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, input):
        eps = self.eps
        
        norm = torch.norm(input, dim=2, keepdim=True)
        x = input / (norm + eps)
        x_U = self.conv1d_U(x)
        x_V = self.conv1d_V(x)
        output = x_U * x_V
        
        return output

class Separator(nn.Module):
    """
    Default separator of TasNet.
    """
    def __init__(self, n_bases, num_blocks, num_layers, hidden_channels, causal=False, n_sources=2, eps=EPS):
        super().__init__()
        
        self.num_blocks = num_blocks
        self.n_bases, self.n_sources = n_bases, n_sources
        self.eps = eps
        
        hidden_channels = n_sources*n_bases
        
        self.gamma = nn.Parameter(torch.Tensor(1, n_bases, 1))
        self.beta = nn.Parameter(torch.Tensor(1, n_bases, 1))
        
        net = []
        
        for idx in range(num_blocks):
            if idx == 0:
                net.append(LSTMBlock(n_bases, n_sources*n_bases, hidden_channels=hidden_channels, num_layers=num_layers, causal=causal))
            else:
                net.append(LSTMBlock(n_sources*n_bases, n_sources*n_bases, hidden_channels=hidden_channels, num_layers=num_layers))
            
        self.net = nn.Sequential(*net)
            
        self._reset_parameters()
                
    def _reset_parameters(self):
        self.gamma.data.fill_(1)
        self.beta.data.zero_()
        
    def forward(self, input):
        num_blocks = self.num_blocks
        n_bases, n_sources = self.n_bases, self.n_sources
        eps = self.eps
        
        batch_size, _, T_bin = input.size()
    
        mean = input.mean(dim=1, keepdim=True)
        squared_mean = (input**2).mean(dim=1, keepdim=True)
        var = squared_mean - mean**2
        x = self.gamma * (input - mean)/(torch.sqrt(var) + eps) + self.beta
        
        skip = 0
        
        for idx in range(num_blocks):
            x = self.net[idx](x)
            skip = x + skip
            
        x = skip
        output = x.view(batch_size, n_sources, n_bases, T_bin)

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
        x = input.permute(0,2,1) # -> (batch_size, T, in_channels)
        x, (_, _) = self.rnn(x) # -> (batch_size, T, out_channels//num_directions)
        x = self.fc(x) # -> (batch_size, T, out_channels)
        output = x.permute(0,2,1) # -> (batch_size, out_channels, T)
          
        return output

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    
    torch.manual_seed(111)
    
    batch_size = 2
    C = 1
    T = 64
    kernel_size, stride = 8, 2
    repeat = 2
    in_channels, n_bases = 1, kernel_size * repeat * 2
    
    input = torch.randn((batch_size, C, T), dtype=torch.float)
    
    window_fn = 'hamming'
    
    model = TasNetBase(in_channels, n_bases, kernel_size=kernel_size, stride=stride, window_fn=window_fn)
    output = model(input)
    print(input.size(), output.size())

    plt.figure()
    plt.plot(range(T), input[0,0].numpy())
    plt.plot(range(T), output[0,0].numpy())
    plt.savefig('data/Fourier.png', bbox_inches='tight')
    plt.close()
    
    bases = model.decoder.get_bases()
    print(bases.shape)
    
    plt.figure()
    plt.pcolormesh(bases, cmap='bwr', norm=Normalize(vmin=-1, vmax=1))
    plt.colorbar()
    plt.savefig('data/bases.png', bbox_inches='tight')
    plt.close()

    _, latent = model.extract_latent(input)
    print(latent.size())
    real = latent[:,:n_bases//2,:]
    imag = latent[:,n_bases//2:,:]
    power = real**2+imag**2
    
    plt.figure()
    plt.pcolormesh(power[0], cmap='bwr')
    plt.colorbar()
    plt.savefig('data/power.png', bbox_inches='tight')
    plt.close()
    
    print("="*10, "LSTM-TasNet", "="*10)
    # LSTM-TasNet configuration
    sep_num_blocks, sep_num_layers, sep_hidden_channels = 2, 2, 32
    n_sources = 3
    
    # Non causal
    print("-"*10, "Non causal", "-"*10)
    causal = False

    model = TasNet(C, n_bases, kernel_size=kernel_size, stride=stride, sep_num_blocks=sep_num_blocks, sep_num_layers=sep_num_layers, sep_hidden_channels=sep_hidden_channels, causal=causal, n_sources=n_sources)
    print(model)
    print("# Parameters: {}".format(model.num_parameters))
    
    output = model(input)
    print(input.size(), output.size())
    print()
    
    # Causal
    print("-"*10, "Causal", "-"*10)
    causal = True
    
    model = TasNet(C, n_bases, kernel_size=kernel_size, stride=stride, sep_num_blocks=sep_num_blocks, sep_num_layers=sep_num_layers, sep_hidden_channels=sep_hidden_channels, causal=causal, n_sources=n_sources)
    print(model)
    print("# Parameters: {}".format(model.num_parameters))

    output = model(input)
    print(input.size(), output.size())
    print()

    print("="*10, "Encoder and pseudo inverse of encoder", "="*10)
    encoder = Encoder(C, n_bases, kernel_size, stride=stride)
    decoder = PinvEncoder(encoder)
    latent = encoder(input)
    output = decoder(latent)
    print(input.size(), output.size())

    plt.figure()
    plt.plot(range(T), input[0,0].detach().numpy())
    plt.plot(range(T), output[0,0].detach().numpy())
    plt.savefig('data/pinverse.png', bbox_inches='tight')
    plt.close()
