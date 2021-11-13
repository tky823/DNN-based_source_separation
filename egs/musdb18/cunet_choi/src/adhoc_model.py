import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from utils.cunet import choose_nonlinear
from models.film import FiLM2d
from models.pocm import PoCM2d, GPoCM2d
from models.cunet import TDF2d, TDC2d

EPS = 1e-12

class TDFUNet2d(nn.Module):
    def __init__(
            self,
            channels,
            kernel_size, stride=None,
            num_layers=2,
            bias=False,
            enc_nonlinear='relu', dec_nonlinear='relu',
            out_channels=None,
            conditioning='film',
            eps=EPS
        ):
        """
        Args:
            channels <list<int>>:
        """
        super().__init__()
        
        raise NotImplementedError("In progress")

class TDFEncoder2d(nn.Module):
    def __init__(self, channels, kernel_size, stride=None, num_layers=2, bias=False, nonlinear='relu', conditioning='film', eps=EPS):
        """
        Args:
            channels <list<int>>
            kernel_size <tuple<int,int>> or <list<tuple<int,int>>>
            stride <tuple<int,int>> or <list<tuple<int,int>>>
            num_layers <int> or <list<int>>
            nonlinear <str> or <list<str>>
        """
        super().__init__()
        
        raise NotImplementedError("In progress")

class TDFDecoder2d(nn.Module):
    def __init__(self, channels, kernel_size, stride=None, num_layers=2, bias=False, nonlinear='relu', eps=EPS):
        """
        Args:
            channels <list<int>>
            kernel_size <tuple<int,int>> or <list<tuple<int,int>>>
            stride <tuple<int,int>> or <list<tuple<int,int>>>
            num_layers <int> or <list<int>>
            nonlinear <str> or <list<str>>
        """
        super().__init__()
        
        raise NotImplementedError("In progress")

class TDFEncoderBlock2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_layers=2, bias=False, nonlinear='relu', resample='conv', conditioning='film', eps=EPS, **kwargs):
        super().__init__()

        raise NotImplementedError("In progress")

class TDFDecoderBlock2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_layers=2, bias=False, nonlinear='relu', resample='conv', eps=EPS, **kwargs):
        super().__init__()

        raise NotImplementedError("In progress")

class TDCUNet2d(nn.Module):
    def __init__(
            self,
            channels,
            kernel_size, stride=None,
            num_layers=2,
            bias=False,
            enc_nonlinear='relu', dec_nonlinear='relu',
            out_channels=None,
            conditioning='film',
            eps=EPS
        ):
        """
        Args:
            channels <list<int>>:
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

        self.preprocess = PreprocessBlock(enc_channels[0], enc_channels[1])
        self.encoder = TDCEncoder2d(enc_channels[1:], kernel_size=kernel_size, stride=stride, num_layers=num_layers, bias=bias, nonlinear=enc_nonlinear, conditioning=conditioning, eps=eps)
        self.bottleneck = nn.Conv2d(channels[-1], channels[-1], kernel_size=(1,1), stride=(1,1))
        self.decoder = TDCDecoder2d(dec_channels[:-1], kernel_size=kernel_size, stride=stride, num_layers=num_layers, bias=bias, nonlinear=dec_nonlinear, eps=eps)
        self.postprocess = PostprocessBlock(dec_channels[-2] // 2, dec_channels[-1])
        
        self.channels = channels
        self.kernel_size, self.stride = kernel_size, stride
        self.bias = bias
        self.enc_nonlinear, self.dec_nonlinear = enc_nonlinear, dec_nonlinear
        self.out_channels = out_channels
        self.conditioning = conditioning
        self.eps = eps

    def forward(self, input, gamma, beta):
        x = self.preprocess(input)
        x, skip = self.encoder(x, gamma, beta)
        x = self.bottleneck(x)
        x = self.decoder(x, skip[::-1])
        output = self.postprocess(x)
        
        return output
    
    @classmethod
    def build_from_config(cls, config):
        channels = config['channels']
        kernel_size, stride = config['kernel_size'], config['stride']
        enc_nonlinear, dec_nonlinear = config['enc_nonlinear'], config['dec_nonlinear']
        bias = config['bias']
        out_channels = config['out_channels']
        conditioning = config['conditioning']
        
        model = cls(
            channels, kernel_size=kernel_size, stride=stride, enc_nonlinear=enc_nonlinear, dec_nonlinear=dec_nonlinear,
            bias=bias,
            out_channels=out_channels,
            conditioning=conditioning
        )
    
        return model

    def get_config(self):
        config = {
            'channels': self.channels,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'enc_nonlinear': self.enc_nonlinear, 'dec_nonlinear': self.dec_nonlinear,
            'out_channels': self.out_channels,
            'conditioning': self.conditioning,
            'eps': self.eps
        }

        return config

class TDCEncoder2d(nn.Module):
    def __init__(self, channels, kernel_size, stride=None, num_layers=2, bias=False, nonlinear='relu', conditioning='film', eps=EPS):
        """
        Args:
            channels <list<int>>
            kernel_size <tuple<int,int>> or <list<tuple<int,int>>>
            stride <tuple<int,int>> or <list<tuple<int,int>>>
            num_layers <int> or <list<int>>
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
        if type(num_layers) is not list:
            num_layers = [num_layers] * n_blocks
        if type(nonlinear) is not list:
            nonlinear = [nonlinear] * n_blocks
        
        self.channels = channels
        self.n_blocks = n_blocks
        
        net = []
        
        for n in range(n_blocks):
            net.append(TDCEncoderBlock2d(channels[n], channels[n + 1], kernel_size=kernel_size[n], stride=stride[n], num_layers=num_layers[n], bias=bias, nonlinear=nonlinear[n], conditioning=conditioning, eps=eps))
        
        self.net = nn.Sequential(*net)
        
    def forward(self, input, gamma, beta):
        n_blocks = self.n_blocks
        
        x = input
        skip = []
        
        for n in range(n_blocks):
            x = self.net[n](x, gamma[n], beta[n])
            skip.append(x)
        
        return x, skip

class TDCDecoder2d(nn.Module):
    def __init__(self, channels, kernel_size, stride=None, num_layers=2, bias=False, nonlinear='relu', eps=EPS):
        """
        Args:
            channels <list<int>>
            kernel_size <tuple<int,int>> or <list<tuple<int,int>>>
            stride <tuple<int,int>> or <list<tuple<int,int>>>
            num_layers <int> or <list<int>>
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
        if type(num_layers) is not list:
            num_layers = [num_layers] * n_blocks
        if type(nonlinear) is not list:
            nonlinear = [nonlinear] * n_blocks
            
        self.n_blocks = n_blocks
        
        net = []
        
        for n in range(n_blocks):
            net.append(TDCDecoderBlock2d(channels[n], channels[n + 1] // 2, kernel_size=kernel_size[n], stride=stride[n], num_layers=num_layers[n], bias=bias, nonlinear=nonlinear[n], eps=eps))
            # channels[n + 1] // 2: because of skip connection
        
        self.net = nn.Sequential(*net)
            
    def forward(self, input, skip):
        """
        Args:
            input (batch_size, C1, H, W)
            skip <list<torch.Tensor>>
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

class TDCEncoderBlock2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_layers=2, bias=False, nonlinear='relu', resample='conv', conditioning='film', eps=EPS, **kwargs):
        super().__init__()

        self.kernel_size = kernel_size

        self.tdc2d = TDC2d(in_channels, out_channels, kernel_size=kernel_size, num_layers=num_layers, nonlinear=nonlinear, bias=bias, eps=eps)

        down_scale = kwargs.get('down_scale') or 2
        down_scale = _pair(down_scale)

        if resample == 'down':
            down_kernel_size = down_scale
            self.downsample2d = nn.AvgPool2d(kernel_size=down_scale, stride=down_scale)
        elif resample == 'conv':
            down_kernel_size = kwargs.get('down_kernel_size') or down_scale
            self.downsample2d = nn.Conv2d(out_channels, out_channels, kernel_size=down_kernel_size, stride=down_scale)
        else:
            raise ValueError()
        
        down_kernel_size = _pair(down_kernel_size)
        self.down_kernel_size, self.down_scale = down_kernel_size, down_scale

        if conditioning == 'film':
            self.conditioning = FiLM2d()
        elif conditioning == 'pocm':
            self.conditioning = PoCM2d()
        elif conditioning == 'gpocm':
            self.conditioning = GPoCM2d()
        else:
            raise ValueError("Not support conditioning {}".format(conditioning))
            
    def forward(self, input, gamma, beta):
        """
        Args:
            input (batch_size, C, H, W)
        """
        down_kernel_size, down_scale = self.down_kernel_size, self.down_scale
        Kh, Kw = down_kernel_size
        Sh, Sw = down_scale
        padding_height = Kh - Sh
        padding_width = Kw - Sw
        padding_top = padding_height // 2
        padding_left = padding_width // 2
        padding_bottom = padding_height - padding_top
        padding_right = padding_width - padding_left

        x = self.tdc2d(input)
        x = F.pad(x, (padding_left, padding_right, padding_top, padding_bottom))
        x = self.downsample2d(x)
        output = self.conditioning(x, gamma, beta)
        
        return output

class TDCDecoderBlock2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_layers=2, bias=False, nonlinear='relu', resample='conv', eps=EPS, **kwargs):
        super().__init__()

        self.kernel_size = kernel_size

        self.tdc2d = TDC2d(in_channels, out_channels, kernel_size=kernel_size, num_layers=num_layers, nonlinear=nonlinear, bias=bias, eps=eps)

        up_scale = kwargs.get('up_scale') or 2
        up_scale = _pair(up_scale)

        if resample == 'up':
            up_kernel_size = up_scale
            self.upsample2d = nn.Upsample(scale_factor=up_scale)
        elif resample == 'conv':
            up_kernel_size = kwargs.get('up_kernel_size') or up_scale
            self.upsample2d = nn.Conv2d(out_channels, out_channels, kernel_size=up_kernel_size, stride=up_scale)
        else:
            raise ValueError()
        
        up_kernel_size = _pair(up_kernel_size)
        self.up_kernel_size, self.up_scale = up_kernel_size, up_scale
    
    def forward(self, input, skip=None):
        """
        Args:
            input (batch_size, C1, H, W)
            skip (batch_size, C2, H, W)
        Returns:
            output: (batch_size, C, H_out, W_out)
        """
        if skip is not None:
            _, _, H_in, W_in = input.size()
            _, _, H_skip, W_skip = skip.size()
            padding_height = H_in - H_skip
            padding_width = W_in - W_skip
            padding_top = padding_height // 2
            padding_bottom = padding_height - padding_top
            padding_left = padding_width // 2
            padding_right = padding_width - padding_left
            
            input = F.pad(input, (-padding_left, -padding_right, -padding_top, -padding_bottom))
            input = torch.cat([input, skip], dim=1)
        
        x = self.tdc2d(input)

        up_kernel_size, up_scale = self.up_kernel_size, self.up_scale
        Kh, Kw = up_kernel_size
        Sh, Sw = up_scale
        padding_height = Kh - Sh
        padding_width = Kw - Sw
        padding_top = padding_height // 2
        padding_left = padding_width // 2
        padding_bottom = padding_height - padding_top
        padding_right = padding_width - padding_left

        x = self.upsample2d(x)
        output = F.pad(x, (-padding_left, -padding_right, -padding_top, -padding_bottom))
        
        return output

class PreprocessBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(2,1), nonlinear='relu', eps=EPS):
        super().__init__()

        self.kernel_size = _pair(kernel_size)
        self.nonlinear = nonlinear

        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride=(1,1))
        self.norm2d = nn.BatchNorm2d(out_channels, eps=eps)

        if self.nonlinear:
            self.nonlinear2d = choose_nonlinear(nonlinear)
    
    def forward(self, input):
        Kh, Kw = self.kernel_size
        padding_height, padding_width = Kh - 1, Kw - 1
        padding_top, padding_left = padding_height // 2, padding_width // 2
        padding_bottom, padding_right = padding_height - padding_top, padding_width - padding_left
        
        x = F.pad(input, (padding_left, padding_right, padding_top, padding_bottom))
        x = self.conv2d(x)
        x = self.norm2d(x)

        if self.nonlinear:
            output = self.nonlinear2d(x)
        else:
            output = x

        return output

class PostprocessBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(2,1), nonlinear=None, eps=EPS):
        super().__init__()

        self.kernel_size = _pair(kernel_size)
        self.nonlinear = nonlinear

        self.deconv2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=(1,1))
        self.norm2d = nn.BatchNorm2d(out_channels, eps=eps)

        if self.nonlinear:
            self.nonlinear2d = choose_nonlinear(nonlinear)
    
    def forward(self, input):
        Kh, Kw = self.kernel_size
        padding_height, padding_width = Kh - 1, Kw - 1
        padding_top, padding_left = padding_height // 2, padding_width // 2
        padding_bottom, padding_right = padding_height - padding_top, padding_width - padding_left
        
        x = self.deconv2d(input)
        x = F.pad(x, (padding_left, padding_right, padding_top, padding_bottom))
        x = self.norm2d(x)

        if self.nonlinear:
            output = self.nonlinear2d(x)
        else:
            output = x
        
        return output

def _test_tdc_cunet():
    latent_dim = 4
    n_bins, n_frames = 512, 128

    channels = [1, 5, 10, 15, 20]
    kernel_size = 3
    stride = 2
    
    dec_nonlinear = ['relu', 'relu', 'relu', 'relu']
    dropout_control = 0.5

    batch_size = 2

    input = torch.randn((batch_size, 1, n_bins, n_frames), dtype=torch.float)

    print('-'*10, "with Complex FiLM and Dense control", '-'*10)
    channels_control = [latent_dim, 4, 8, 16]
    out_channels_control = [1, 1, 1, 1]

    input_latent = torch.randn((batch_size, latent_dim), dtype=torch.float)

    control_net = ControlDenseNet(channels_control, out_channels_control, nonlinear=False, dropout=dropout_control, norm=True)
    unet = TDCUNet2d(channels, kernel_size=kernel_size, stride=stride, dec_nonlinear=dec_nonlinear, conditioning='film')
    model = ConditionedUNet2d(control_net=control_net, unet=unet)
    output = model(input, input_latent)
    print(model)
    print(input.size(), input_latent.size(), output.size())

if __name__ == '__main__':
    from models.cunet import ControlDenseNet
    from models.cunet import ConditionedUNet2d

    _test_tdc_cunet()
    print()