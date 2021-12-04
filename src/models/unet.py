import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from utils.model import choose_nonlinear
from conv import DepthwiseSeparableConv1d, DepthwiseSeparableConvTranspose1d, DepthwiseSeparableConv2d, DepthwiseSeparableConvTranspose2d

class UNetBase(nn.Module):
    def __init__(self):
        super().__init__()
        
    @classmethod
    def build_model(cls, model_path, load_state_dict=False):
        config = torch.load(model_path, map_location=lambda storage, loc: storage)
        
        channels = config['channels']
        kernel_size, stride, dilated = config['kernel_size'], config['stride'], config['dilated']
        enc_nonlinear, dec_nonlinear = config.get('enc_nonlinear') or config['nonlinear_enc'], config.get('dec_nonlinear') or config['nonlinear_dec']
        out_channels = config['out_channels']
        
        model = cls(channels, kernel_size, stride=stride, dilated=dilated, enc_nonlinear=enc_nonlinear, dec_nonlinear=dec_nonlinear, out_channels=out_channels)
        
        if load_state_dict:
            model.load_state_dict(config['state_dict'])
        
        return model
    
    def get_config(self):
        config = {
            'channels': self.channels,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'dilated': self.dilated,
            'enc_nonlinear': self.enc_nonlinear,
            'dec_nonlinear': self.dec_nonlinear,
            'out_channels': self.out_channels
        }
        
        return config
    
    @property
    def num_parameters(self):
        _num_parameters = 0
        
        for p in self.parameters():
            if p.requires_grad:
                _num_parameters += p.numel()
                
        return _num_parameters

class UNet1d(UNetBase):
    def __init__(self, channels, kernel_size, stride=None, dilated=False, enc_nonlinear='relu', dec_nonlinear='relu', out_channels=None):
        """
        Args:
            channels <list<int>>
            out_channels <int>
        """
        super().__init__()
        
        enc_channels = channels
        
        if out_channels is None:
            dec_channels = channels[::-1]
        else:
            dec_channels = channels[:0:-1] + [out_channels]
            
        _dec_channels = []
        for idx, out_channel in enumerate(dec_channels):
            if idx == 0:
                _dec_channels.append(out_channel)
            else:
                _dec_channels.append(2 * out_channel)
        dec_channels = _dec_channels
        
        self.channels = channels
        self.kernel_size, self.stride, self.dilated = kernel_size, stride, dilated
        self.enc_nonlinear, self.dec_nonlinear = enc_nonlinear, dec_nonlinear
        self.out_channels = out_channels

        self.encoder = Encoder1d(enc_channels, kernel_size=kernel_size, stride=stride, dilated=dilated, nonlinear=enc_nonlinear)
        self.bottleneck = nn.Conv1d(channels[-1], channels[-1], kernel_size=1, stride=1)
        self.decoder = Decoder1d(dec_channels, kernel_size=kernel_size, stride=stride, dilated=dilated, nonlinear=dec_nonlinear)
        
    def forward(self, input):
        x, skip = self.encoder(input)
        x = self.bottleneck(x)
        output = self.decoder(x, skip[::-1])
        
        return output

class UNet2d(UNetBase):
    def __init__(self, channels, kernel_size, stride=None, dilated=False, enc_nonlinear='relu', dec_nonlinear='relu', out_channels=None):
        """
        Args:
            channels <list<int>>
            out_channels <int>
        """
        super().__init__()
        
        enc_channels = channels
        
        if out_channels is None:
            dec_channels = channels[::-1]
        else:
            dec_channels = channels[:0:-1] + [out_channels]
            
        _dec_channels = []
        for idx, out_channel in enumerate(dec_channels):
            if idx == 0:
                _dec_channels.append(out_channel)
            else:
                _dec_channels.append(2 * out_channel)
        dec_channels = _dec_channels
        
        self.channels = channels
        self.kernel_size, self.stride, self.dilated = kernel_size, stride, dilated
        self.enc_nonlinear, self.dec_nonlinear = enc_nonlinear, dec_nonlinear
        self.out_channels = out_channels

        self.encoder = Encoder2d(enc_channels, kernel_size=kernel_size, stride=stride, dilated=dilated, nonlinear=enc_nonlinear)
        self.bottleneck = nn.Conv2d(channels[-1], channels[-1], kernel_size=(1,1), stride=(1,1))
        self.decoder = Decoder2d(dec_channels, kernel_size=kernel_size, stride=stride, dilated=dilated, nonlinear=dec_nonlinear)
        
    def forward(self, input):
        x, skip = self.encoder(input)
        x = self.bottleneck(x)
        output = self.decoder(x, skip[::-1])
        
        return output

"""
    Encoder
"""
class Encoder1d(nn.Module):
    def __init__(self, channels, kernel_size, stride=None, dilated=False, separable=False, nonlinear='relu'):
        """
        Args:
            channels <list<int>>
            kernel_size <int> or <list<int>>
            stride <int> or <list<int>>
            dilated <bool>
            nonlinear <str> or <list<str>>
        """
        super().__init__()
        
        n_blocks = len(channels) - 1
        
        if type(kernel_size) is not list:
            kernel_size = [kernel_size] * n_blocks
        if stride is None:
            stride = kernel_size
        elif type(stride) is not list:
            stride = [stride] * n_blocks
        if type(nonlinear) is not list:
            nonlinear = [nonlinear] * n_blocks
        
        self.n_blocks = n_blocks
        
        net = []
        
        for n in range(n_blocks):
            if dilated:
                dilation = 2**n
                assert stride[n] == 1, "stride must be 1 when dilated convolution."
            else:
                dilation = 1
            net.append(EncoderBlock1d(channels[n], channels[n+1], kernel_size=kernel_size[n], stride=stride[n], dilation=dilation, separable=separable, nonlinear=nonlinear))
        
        self.net = nn.Sequential(*net)
        
    def forward(self, input):
        n_blocks = self.n_blocks
        
        x = input
        skip = []
        
        for n in range(n_blocks):
            x = self.net[n](x)
            skip.append(x)
        
        return x, skip

class Encoder2d(nn.Module):
    def __init__(self, channels, kernel_size, stride=None, dilated=False, separable=False, nonlinear='relu'):
        """
        Args:
            channels <list<int>>
            kernel_size <tuple<int,int>> or <list<tuple<int,int>>>
            stride <tuple<int,int>> or <list<tuple<int,int>>>
            dilated <bool>
            nonlinear <str> or <list<str>>
        """
        super().__init__()
        
        n_blocks = len(channels) - 1
        
        if type(kernel_size) is not list:
            kernel_size = _pair(kernel_size)
            kernel_size = [kernel_size] * n_blocks
        if stride is None:
            stride = kernel_size
        elif type(stride) is not list:
            stride = [stride] * n_blocks
        if type(nonlinear) is not list:
            nonlinear = [nonlinear] * n_blocks
        
        self.n_blocks = n_blocks
        
        net = []
        
        for n in range(n_blocks):
            if dilated:
                dilation = 2**n
                assert stride[n] == 1, "stride must be 1 when dilated convolution."
            else:
                dilation = 1
            
            net.append(EncoderBlock2d(channels[n], channels[n + 1], kernel_size=kernel_size[n], stride=stride[n], dilation=dilation, separable=separable, nonlinear=nonlinear[n]))
        
        self.net = nn.Sequential(*net)
        
    def forward(self, input):
        n_blocks = self.n_blocks
        
        x = input
        skip = []
        
        for n in range(n_blocks):
            x = self.net[n](x)
            skip.append(x)
        
        return x, skip

"""
    Decoder
"""
class Decoder1d(nn.Module):
    def __init__(self, channels, kernel_size, stride=None, dilated=False, separable=False, nonlinear='relu'):
        """
        Args:
            channels <list<int>>
            kernel_size <int> or <list<int>>
            stride <int> or <list<int>>
            dilated <bool>
            nonlinear <str> or <list<str>>
        """
        super().__init__()
        
        n_blocks = len(channels) - 1
        
        if type(kernel_size) is not list:
            kernel_size = [kernel_size] * n_blocks
        if stride is None:
            stride = kernel_size
        elif type(stride) is not list:
            stride = [stride] * n_blocks
        if type(nonlinear) is not list:
            nonlinear = [nonlinear] * n_blocks
            
        self.n_blocks = n_blocks
        
        net = []
        
        for n in range(n_blocks):
            if dilated:
                dilation = 2**(n_blocks - n - 1)
                assert stride[n] == 1, "stride must be 1 when dilated convolution."
            else:
                dilation = 1
            net.append(DecoderBlock1d(channels[n], channels[n + 1] // 2, kernel_size=kernel_size[n], stride=stride[n], dilation=dilation, separable=separable, nonlinear=nonlinear[n]))
            # channels[n + 1] // 2: because of skip connection
    
        self.net = nn.Sequential(*net)
            
    def forward(self, input, skip):
        """
        Args:
            input (batch_size, C, T)
            skip <list<torch.Tensor>>
        Returns:
            output: (batch_size, C_out, T_out)
        """
        n_blocks = self.n_blocks
        
        x = input
        
        for n in range(n_blocks):
            if n == 0:
                x = self.net[n](x)
            else:
                x = self.net[n](x, skip[n])
        output = x
        
        return output

class Decoder2d(nn.Module):
    def __init__(self, channels, kernel_size, stride=None, dilated=False, separable=False, nonlinear='relu'):
        """
        Args:
            channels <list<int>>
            kernel_size <tuple<int,int>> or <list<tuple<int,int>>>
            stride <tuple<int,int>> or <list<tuple<int,int>>>
            dilated <bool>
            nonlinear <str> or <list<str>>
        """
        super().__init__()
        
        n_blocks = len(channels) - 1
        
        if type(kernel_size) is not list:
            kernel_size = [kernel_size] * n_blocks
        if stride is None:
            stride = kernel_size
        elif type(stride) is not list:
            stride = [stride] * n_blocks
        if type(nonlinear) is not list:
            nonlinear = [nonlinear] * n_blocks
            
        self.n_blocks = n_blocks
        
        net = []
        
        for n in range(n_blocks):
            if dilated:
                dilation = 2**(n_blocks - n - 1)
                assert stride[n] == 1, "stride must be 1 when dilated convolution."
            else:
                dilation = 1
            
            net.append(DecoderBlock2d(channels[n], channels[n + 1] // 2, kernel_size=kernel_size[n], stride=stride[n], dilation=dilation, separable=separable, nonlinear=nonlinear[n]))
            # channels[n + 1] // 2: because of skip connection
        
        self.net = nn.Sequential(*net)
            
    def forward(self, input, skip):
        """
        Args:
            input (batch_size, C, H, W)
            skip <list<torch.Tensor>>
        Returns:
            output: (batch_size, C_out, H_out, W_out)
        """
        n_blocks = self.n_blocks
        
        x = input
        
        for n in range(n_blocks):
            if n == 0:
                x = self.net[n](x)
            else:
                x = self.net[n](x, skip[n])
        
        output = x
        
        return output

"""
    Encoder Block
"""
class EncoderBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=None, dilation=1, separable=False, nonlinear='relu'):
        super().__init__()

        if stride is None:
            stride = kernel_size
    
        self.kernel_size, self.stride, self.dilation = kernel_size, stride, dilation

        if separable:
            self.conv1d = DepthwiseSeparableConv1d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation)
        else:
            self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation)
        
        self.norm1d = nn.BatchNorm1d(out_channels)

        if nonlinear == "softmax":
            kwargs = {
                "dim": 1
            }
        else:
            kwargs = {}
        
        self.nonlinear = choose_nonlinear(nonlinear, **kwargs)
            
    def forward(self, input):
        """
        Args:
            input (batch_size, C, T)
        """
        kernel_size, stride, dilation = self.kernel_size, self.stride, self.dilation
        kernel_size = (kernel_size - 1) * dilation + 1
        
        _, _, T = input.size()
        padding = kernel_size - 1 - (stride - (T - kernel_size) % stride) % stride
        padding_left = padding // 2
        padding_right = padding - padding_left
        
        input = F.pad(input, (padding_left, padding_right))
        
        x = self.conv1d(input)
        x = self.norm1d(x)
        output = self.nonlinear(x)
        
        return output

class EncoderBlock2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=None, dilation=1, separable=False, nonlinear='relu'):
        super().__init__()
        
        kernel_size = _pair(kernel_size)
        
        if stride is None:
            stride = kernel_size
        stride = _pair(stride)
        dilation = _pair(dilation)

        self.kernel_size, self.stride, self.dilation = kernel_size, stride, dilation

        if separable:
            self.conv2d = DepthwiseSeparableConv2d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation)
        else:
            self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation)
        
        self.norm2d = nn.BatchNorm2d(out_channels)
        
        if nonlinear == "softmax":
            kwargs = {
                "dim": 1
            }
        else:
            kwargs = {}
        
        self.nonlinear = choose_nonlinear(nonlinear, **kwargs)
            
    def forward(self, input):
        """
        Args:
            input (batch_size, C, H, W)
        Returns:
            output: (batch_size, C, H_out, W_out), where H_out = H // Sh
        """
        Kh, Kw = self.kernel_size
        Sh, Sw = self.stride
        Dh, Dw = self.dilation
        Kh = (Kh - 1) * Dh + 1
        Kw = (Kw - 1) * Dw + 1
        
        _, _, H, W = input.size()
        Ph, Pw = Kh - 1 - (Sh - (H - Kh) % Sh) % Sh, Kw - 1 - (Sw - (W - Kw) % Sw) % Sw
        Ph_top, Pw_left = Ph // 2, Pw // 2
        Ph_bottom, Pw_right = Ph - Ph_top, Pw - Pw_left
        
        input = F.pad(input, (Pw_left, Pw_right, Ph_top, Ph_bottom))
        
        x = self.conv2d(input)
        x = self.norm2d(x)
        output = self.nonlinear(x)
        
        return output

"""
    Decoder Block
"""
class DecoderBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=None, dilation=1, separable=False, nonlinear='relu'):
        super().__init__()
        
        if stride is None:
            stride = kernel_size

        self.kernel_size, self.stride, self.dilation = kernel_size, stride, dilation

        if separable:
            self.deconv1d = DepthwiseSeparableConvTranspose1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation)
        else:
            self.deconv1d = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation)
        
        self.norm1d = nn.BatchNorm1d(out_channels)

        if nonlinear == "softmax":
            kwargs = {
                "dim": 1
            }
        else:
            kwargs = {}
        
        self.nonlinear = choose_nonlinear(nonlinear, **kwargs)
        
    def forward(self, input, skip=None):
        """
        Args:
            input (batch_size, C1, T)
            skip (batch_size, C2, T)
        Returns:
            output: (batch_size, C, T_out)
        """
        kernel_size, stride, dilation = self.kernel_size, self.stride, self.dilation
        kernel_size = (kernel_size - 1) * dilation + 1
        
        P = kernel_size - stride
        P_left = P // 2
        P_right = P - P_left

        if skip is not None:
            input = self.pad_concat(input, skip)
        
        x = self.deconv1d(input)
        x = F.pad(x, (-P_left, -P_right))
        x = self.norm1d(x)
        output = self.nonlinear(x)
        
        return output

    def pad_concat(self, input, skip):
        """
        Args:
            input (batch_size, C1, T_in)
            skip (batch_size, C2, T_skip)
        Returns:
            output: (batch_size, C, T_skip), where C = C1 + C2
        """
        T_in, T_skip = input.size(-1), skip.size(-1)
        T_padding = T_skip - T_in
        P_left = T_padding // 2
        P_right = T_padding - P_left
        input = F.pad(input, (P_left, P_right))
        output = torch.cat([input, skip], dim=1)

        return output

class DecoderBlock2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=None, dilation=1, separable=False, nonlinear='relu'):
        super().__init__()
        
        kernel_size = _pair(kernel_size)
        
        if stride is None:
            stride = kernel_size
        
        stride = _pair(stride)
        dilation = _pair(dilation)

        self.kernel_size, self.stride, self.dilation = kernel_size, stride, dilation

        if separable:
            self.deconv2d = DepthwiseSeparableConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation)
        else:
            self.deconv2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation)
        
        self.norm2d = nn.BatchNorm2d(out_channels)

        if nonlinear == "softmax":
            kwargs = {
                "dim": 1
            }
        else:
            kwargs = {}
        
        self.nonlinear = choose_nonlinear(nonlinear, **kwargs)
    
    def forward(self, input, skip=None):
        """
        Args:
            input (batch_size, C1, H, W)
            skip (batch_size, C2, H, W)
        Returns:
            output: (batch_size, C, H_out, W_out)
        """
        (Kh, Kw), (Sh, Sw) = self.kernel_size, self.stride
        Dh, Dw = self.dilation
        Kh, Kw = (Kh - 1) * Dh + 1, (Kw - 1) * Dw + 1
        
        Ph, Pw = Kh - Sh, Kw - Sw
        Ph_top, Pw_left = Ph // 2, Pw // 2
        Ph_bottom, Pw_right = Ph - Ph_top, Pw - Pw_left

        if skip is not None:
            input = self.pad_concat(input, skip)

        x = self.deconv2d(input)
        x = F.pad(x, (-Pw_left, -Pw_right, -Ph_top, -Ph_bottom))
        x = self.norm2d(x)
        output = self.nonlinear(x)
        
        return output

    def pad_concat(self, input, skip):
        """
        Args:
            input (batch_size, C1, H_in, W_in)
            skip (batch_size, C2, H_skip, W_skip)
        Returns:
            output: (batch_size, C, H_skip, W_skip), where C = C1 + C2
        """
        (H_in, W_in), (H_skip, W_skip) = input.size()[-2:], skip.size()[-2:]
        Ph, Pw = H_skip - H_in, W_skip - W_in
        Ph_top, Pw_left = Ph // 2, Pw // 2
        Ph_bottom, Pw_right = Ph - Ph_top, Pw - Pw_left
        input = F.pad(input, (Pw_left, Pw_right, Ph_top, Ph_bottom))
        output = torch.cat([input, skip], dim=1)
        
        return output

def _test_unet():
    batch_size = 4
    C = 3
    channels = [C, 8, 16, 16, 32]
    H, W = 512, 256

    kernel_size, stride, dilated = 3, 1, True
    enc_nonlinear = 'relu'
    dec_nonlinear = ['relu', 'relu', 'relu', 'sigmoid']

    input = torch.randint(0, 5, (batch_size, C, H, W), dtype=torch.float)
    print(input.size())

    unet2d = UNet2d(channels, kernel_size=kernel_size, stride=stride, dilated=dilated, enc_nonlinear=enc_nonlinear, dec_nonlinear=dec_nonlinear, out_channels=1)
    print(unet2d)
    print("# Parameters: {}".format(unet2d.num_parameters))

    output = unet2d(input)
    print(output.size())

    package = unet2d.get_package()
    model_path = "unet.pth"
    torch.save(package, model_path)
    _ = UNet2d.build_model(model_path)

if __name__ == '__main__':
    _test_unet()
