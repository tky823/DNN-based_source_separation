import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from utils.cunet import choose_nonlinear, choose_rnn
from conv import DepthwiseSeparableConv1d, DepthwiseSeparableConv2d, DepthwiseSeparableConvTranspose2d
from models.film import FiLM2d
from models.pocm import PoCM2d, GPoCM2d

EPS = 1e-12

"""
Conditioned UNet
    Reference: "Conditioned-U-Net: Introducing a Control Mechanism in the U-Net for multiple source separations"
    See https://arxiv.org/abs/1907.01277
    Reference: "LaSAFT: Latent Source Attentive Frequency Transformation for Conditioned Source Separation"
    See https://arxiv.org/abs/2010.11631
"""

class ConditionedUNetBase(nn.Module):
    def __init__(self):
        super().__init__()

    @property
    def num_parameters(self):
        _num_parameters = 0

        for p in self.parameters():
            if p.requires_grad:
                _num_parameters += p.numel()

        return _num_parameters

class ConditionedUNet2d(ConditionedUNetBase):
    def __init__(
            self,
            control_net,
            unet,
            masking=False,
        ):
        """
        Args:
            channels <list<int>>:
            out_channels <int>:
        """
        super().__init__()

        self.masking = masking

        self.control_net = control_net
        self.backbone = unet

    def forward(self, input, latent):
        gamma, beta = self.control_net(latent)
        x = self.backbone(input, gamma, beta)

        _, _, H_in, W_in = input.size()
        _, _, H, W = x.size()
        padding_height = H - H_in
        padding_width = W - W_in
        padding_top = padding_height // 2
        padding_bottom = padding_height - padding_top
        padding_left = padding_width // 2
        padding_right = padding_width - padding_left

        x = F.pad(x, (-padding_left, -padding_right, -padding_top, -padding_bottom))

        if self.masking:
            output = x * input
        else:
            output = x

        return output

    def get_config(self):
        config = {}
        config['control'] = self.control_net.get_config()
        config['backbone'] = self.backbone.get_config()

        return config

class UNet2d(ConditionedUNetBase):
    def __init__(
            self,
            channels,
            kernel_size, stride=None,
            dilated=False, separable=False, bias=False,
            enc_nonlinear='leaky-relu', dec_nonlinear='leaky-relu',
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

        self.encoder = Encoder2d(enc_channels, kernel_size=kernel_size, stride=stride, dilated=dilated, separable=separable, bias=bias, nonlinear=enc_nonlinear, conditioning=conditioning, eps=eps)
        self.bottleneck = nn.Conv2d(channels[-1], channels[-1], kernel_size=(1,1), stride=(1,1))
        self.decoder = Decoder2d(dec_channels, kernel_size=kernel_size, stride=stride, dilated=dilated, separable=separable, bias=bias, nonlinear=dec_nonlinear, eps=eps)

        self.channels = channels
        self.kernel_size, self.stride = kernel_size, stride
        self.dilated, self.separable = dilated, separable
        self.bias = bias
        self.enc_nonlinear, self.dec_nonlinear = enc_nonlinear, dec_nonlinear
        self.out_channels = out_channels
        self.conditioning = conditioning
        self.eps = eps

    def forward(self, input, gamma, beta):
        x, skip = self.encoder(input, gamma, beta)
        x = self.bottleneck(x)
        output = self.decoder(x, skip[::-1])

        return output

    @classmethod
    def build_from_config(cls, config):
        channels = config['channels']
        kernel_size, stride = config['kernel_size'], config['stride']
        enc_nonlinear, dec_nonlinear = config['enc_nonlinear'], config['dec_nonlinear']
        dilated, separable = config['dilated'], config['separable']
        bias = config['bias']
        out_channels = config['out_channels']
        conditioning = config['conditioning']

        model = cls(
            channels, kernel_size=kernel_size, stride=stride, enc_nonlinear=enc_nonlinear, dec_nonlinear=dec_nonlinear,
            dilated=dilated, separable=separable, bias=bias,
            out_channels=out_channels,
            conditioning=conditioning
        )

        return model

    def get_config(self):
        config = {
            'channels': self.channels,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'dilated': self.dilated,
            'separable': self.separable,
            'enc_nonlinear': self.enc_nonlinear, 'dec_nonlinear': self.dec_nonlinear,
            'out_channels': self.out_channels,
            'conditioning': self.conditioning,
            'eps': self.eps
        }

        return config

class Encoder2d(nn.Module):
    def __init__(self, channels, kernel_size, stride=None, dilated=False, separable=False, bias=False, nonlinear='relu', conditioning='film', eps=EPS):
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

        self.channels = channels
        self.n_blocks = n_blocks

        net = []

        for n in range(n_blocks):
            if dilated:
                dilation = 2**n
                assert stride[n] == 1, "stride must be 1 when dilated convolution."
            else:
                dilation = 1
            net.append(EncoderBlock2d(channels[n], channels[n + 1], kernel_size=kernel_size[n], stride=stride[n], dilation=dilation, separable=separable, bias=bias, nonlinear=nonlinear[n], conditioning=conditioning, eps=eps))

        self.net = nn.Sequential(*net)

    def forward(self, input, gamma, beta):
        n_blocks = self.n_blocks

        x = input
        skip = []

        for n in range(n_blocks):
            x = self.net[n](x, gamma[n], beta[n])
            skip.append(x)

        return x, skip

class Decoder2d(nn.Module):
    def __init__(self, channels, kernel_size, stride=None, dilated=False, separable=False, bias=False, nonlinear='relu', eps=EPS):
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
            net.append(DecoderBlock2d(channels[n], channels[n + 1] // 2, kernel_size=kernel_size[n], stride=stride[n], dilation=dilation, separable=separable, bias=bias, nonlinear=nonlinear[n], eps=eps))
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

class EncoderBlock2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=None, dilation=1, separable=False, bias=False, nonlinear='relu', conditioning='film', eps=EPS):
        super().__init__()

        kernel_size = _pair(kernel_size)

        if stride is None:
            stride = kernel_size
        stride = _pair(stride)
        dilation = _pair(dilation)

        self.kernel_size, self.stride, self.dilation = kernel_size, stride, dilation

        if separable:
            self.conv2d = DepthwiseSeparableConv2d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, bias=bias)
        else:
            self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, bias=bias)

        self.norm2d = nn.BatchNorm2d(out_channels, eps=eps)

        if conditioning == 'film':
            self.conditioning = FiLM2d()
        elif conditioning == 'pocm':
            self.conditioning = PoCM2d()
        elif conditioning == 'gpocm':
            self.conditioning = GPoCM2d()
        else:
            raise ValueError("Not support conditioning {}".format(conditioning))

        self.nonlinear2d = choose_nonlinear(nonlinear)

    def forward(self, input, gamma, beta):
        """
        Args:
            input (batch_size, C, H, W)
        """
        Kh, Kw = self.kernel_size
        Sh, Sw = self.stride
        Dh, Dw = self.dilation

        Kh = (Kh - 1) * Dh + 1
        Kw = (Kw - 1) * Dw + 1

        _, _, H, W = input.size()
        padding_height = Kh - 1 - (Sh - (H - Kh) % Sh) % Sh
        padding_width = Kw - 1 - (Sw - (W - Kw) % Sw) % Sw
        padding_top = padding_height // 2
        padding_bottom = padding_height - padding_top
        padding_left = padding_width // 2
        padding_right = padding_width - padding_left

        input = F.pad(input, (padding_left, padding_right, padding_top, padding_bottom))

        x = self.conv2d(input)
        x = self.norm2d(x)
        x = self.conditioning(x, gamma, beta)
        output = self.nonlinear2d(x)

        return output

class DecoderBlock2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=None, dilation=1, separable=False, bias=False, nonlinear='relu', eps=EPS):
        super().__init__()

        kernel_size = _pair(kernel_size)

        if stride is None:
            stride = kernel_size
        stride = _pair(stride)
        dilation = _pair(dilation)

        self.kernel_size, self.stride, self.dilation = kernel_size, stride, dilation

        if separable:
            self.deconv2d = DepthwiseSeparableConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, bias=bias)
        else:
            self.deconv2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, bias=bias)
        self.norm2d = nn.BatchNorm2d(out_channels, eps=eps)
        self.nonlinear2d = choose_nonlinear(nonlinear)

    def forward(self, input, skip=None):
        """
        Args:
            input (batch_size, C1, H, W)
            skip (batch_size, C2, H, W)
        Returns:
            output: (batch_size, C, H_out, W_out)
        """
        Kh, Kw = self.kernel_size
        Sh, Sw = self.stride
        Dh, Dw = self.dilation
        
        Kh = (Kh - 1) * Dh + 1
        Kw = (Kw - 1) * Dw + 1

        if skip is not None:
            _, _, H_in, W_in = input.size()
            _, _, H_skip, W_skip = skip.size()
            padding_height = H_in - H_skip
            padding_width = W_in - W_skip
            padding_top = padding_height//2
            padding_bottom = padding_height - padding_top
            padding_left = padding_width//2
            padding_right = padding_width - padding_left

            input = F.pad(input, (-padding_left, -padding_right, -padding_top, -padding_bottom))
            input = torch.cat([input, skip], dim=1)

        padding_height = Kh - Sh
        padding_width = Kw - Sw
        padding_top = padding_height//2
        padding_bottom = padding_height - padding_top
        padding_left = padding_width//2
        padding_right = padding_width - padding_left

        x = self.deconv2d(input)
        x = F.pad(x, (-padding_left, -padding_right, -padding_top, -padding_bottom))
        x = self.norm2d(x)
        output = self.nonlinear2d(x)

        return output

class TDF2d(nn.Module):
    """
    Time-Distributed Fully-connected Layer for time-frequency representation
    """
    def __init__(self, num_features, in_bins, out_bins, nonlinear='relu', bias=False, eps=EPS):
        super().__init__()

        self.net = TDFTransformBlock2d(num_features, in_bins, out_bins, nonlinear=nonlinear, bias=bias, eps=eps)

    def forward(self, input):
        output = self.net(input)

        return output

class MultiheadTDF2d(nn.Module):
    def __init__(self, num_features, in_bins, out_bins, num_heads, nonlinear='relu', bias=False, stack_dim=1, eps=EPS):
        super().__init__()

        self.num_heads = num_heads
        self.stack_dim = stack_dim

        net = []

        for idx in range(num_heads):
            net.append(TDFTransformBlock2d(num_features, in_bins, out_bins, nonlinear=nonlinear, bias=bias, eps=eps))

        self.net = nn.ModuleList(net)

    def forward(self, input):
        """
        Args:
            input <torch.Tensor>: (batch_size, in_channels, n_bins, n_frames) if stack_dim=1
        Returns:
            output <torch.Tensor>: (batch_size, num_heads, in_channels, n_bins, n_frames) if stack_dim=1
        """
        output = []

        for idx in range(self.num_heads):
            x = self.net[idx](input)
            output.append(x)

        output = torch.stack(output, dim=self.stack_dim)

        return output

class TDFTransformBlock2d(nn.Module):
    def __init__(self, num_features, in_bins, out_bins, nonlinear='relu', bias=False, eps=EPS):
        super().__init__()

        self.nonlinear = nonlinear

        self.conv1d = nn.Conv1d(in_bins, out_bins, kernel_size=1, stride=1, bias=bias)
        self.norm2d = nn.BatchNorm2d(num_features, eps=eps)

        if nonlinear:
            self.nonlinear2d = choose_nonlinear(nonlinear)

    def forward(self, input):
        """
        Args:
            input <torch.Tensor>: (batch_size, num_features, in_bins, n_frames)
        Returns:
            output <torch.Tensor>: (batch_size, num_features, out_bins, n_frames)
        """
        batch_size, num_features, _, n_frames = input.size()

        x = input.view(batch_size * num_features, -1, n_frames)
        x = self.conv1d(x)
        x = x.view(batch_size, num_features, -1, n_frames)
        x = self.norm2d(x)

        if self.nonlinear:
            output = self.nonlinear2d(x)
        else:
            output = x

        return output

class TFC2d(nn.Module):
    """
    Time-Frequency Convolutions
    """
    def __init__(self, in_channels, growth_rate, kernel_size, num_layers=2, nonlinear='relu', bias=False, eps=EPS):
        super().__init__()

        _in_channels = in_channels

        net = []

        for idx in range(num_layers):
            net.append(TFCTransformBlock2d(_in_channels, growth_rate, kernel_size=kernel_size, stride=(1,1), nonlinear=nonlinear, bias=bias, eps=eps))
            _in_channels += growth_rate

        self.net = nn.Sequential(*net)

        self.num_layers = num_layers

    def forward(self, input):
        stack = input
        for idx in range(self.num_layers):
            x = self.net[idx](stack)
            if idx == self.num_layers - 1:
                output = x
            else:
                stack = torch.cat([stack, x], dim=1)

        return output

class TFCTransformBlock2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1,1), nonlinear='relu', bias=False, eps=EPS):
        super().__init__()

        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.nonlinear = nonlinear

        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, bias=bias)
        self.norm2d = nn.BatchNorm2d(out_channels, eps=eps)

        if nonlinear:
            self.nonlinear2d = choose_nonlinear(nonlinear)

    def forward(self, input):
        Kh, Kw = self.kernel_size
        Sh, Sw = self.stride

        padding_height = Kh - Sh
        padding_width = Kw - Sw
        padding_top = padding_height // 2
        padding_left = padding_width // 2
        padding_bottom = padding_height - padding_top
        padding_right = padding_width - padding_left

        x = F.pad(input, (padding_left, padding_right, padding_top, padding_bottom))
        x = self.conv2d(x)
        x = self.norm2d(x)

        if self.nonlinear:
            output = self.nonlinear2d(x)
        else:
            output = x

        return output

class TDC2d(nn.Module):
    def __init__(self, in_channels, growth_rate, kernel_size, num_layers=2, nonlinear='relu', bias=False, eps=EPS):
        super().__init__()

        _in_channels = in_channels

        net = []

        for idx in range(num_layers):
            net.append(TDCTransformBlock2d(_in_channels, growth_rate, kernel_size=kernel_size, stride=1, nonlinear=nonlinear, bias=bias, eps=eps))
            _in_channels += growth_rate

        self.net = nn.Sequential(*net)

        self.num_layers = num_layers

    def forward(self, input):
        """
        Args:
            input (batch_size, in_channels, n_bins, n_frames)
        Returns:
            input (batch_size, growth_rate, n_bins, n_frames)
        """
        stack = input
        for idx in range(self.num_layers):
            x = self.net[idx](stack)
            if idx == self.num_layers - 1:
                output = x
            else:
                stack = torch.cat([stack, x], dim=1)

        return output

class TDCTransformBlock2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, nonlinear='relu', bias=False, eps=EPS):
        super().__init__()

        self.in_channels, self.out_channels = in_channels, out_channels
        self.kernel_size, self.stride = kernel_size, stride
        self.nonlinear = nonlinear

        self.conv1d = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, bias=bias)
        self.norm1d = nn.BatchNorm1d(out_channels, eps=eps)

        if nonlinear:
            self.nonlinear2d = choose_nonlinear(nonlinear)

    def forward(self, input):
        """
        Args:
            input (batch_size, in_channels, n_bins, n_frames)
        Returns:
            input (batch_size, out_channels, n_bins, n_frames)
        """
        in_channels, out_channels = self.in_channels, self.out_channels
        K, S = self.kernel_size, self.stride
        batch_size, _, n_bins, n_frames = input.size()

        padding = K - S
        padding_left = padding // 2
        padding_right = padding - padding_left

        x = input.permute(0, 3, 1, 2).contiguous() # (batch_size, n_frames, in_channels, n_bins)
        x = x.view(batch_size * n_frames, in_channels, n_bins)
        x = F.pad(x, (padding_left, padding_right))
        x = self.conv1d(x)
        x = self.norm1d(x)

        if self.nonlinear:
            output = self.nonlinear2d(x)
        else:
            output = x

        x = x.view(batch_size, n_frames, out_channels, n_bins)
        output = x.permute(0, 2, 3, 1).contiguous() # (batch_size, out_channels, n_bins, n_frames)

        return output

class TDCRNN2d(nn.Module):
    def __init__(self, in_channels, growth_rate, kernel_size, n_bins, bottleneck_bins=None, hidden_channels=None, num_layers_tdc=2, num_layers_rnn=2, nonlinear='relu', rnn='gru', causal=False, bias=False, eps=EPS):
        super().__init__()

        if causal:
            num_directions = 1
            bidirectional = False
        else:
            num_directions = 2
            bidirectional = True

        self.tdc2d = TDC2d(in_channels, growth_rate, kernel_size=kernel_size, num_layers=num_layers_tdc, nonlinear=nonlinear, bias=bias)
        self.norm2d = nn.BatchNorm2d(growth_rate)
        self.rnn = choose_rnn(rnn, input_size=n_bins, hidden_size=hidden_channels, num_layers=num_layers_rnn, batch_first=True, bidirectional=bidirectional)

        out_channels = num_directions * hidden_channels

        self.tdf2d = nn.Sequential(
            TDF2d(growth_rate, out_channels, bottleneck_bins, nonlinear=nonlinear, bias=bias, eps=eps),
            TDF2d(growth_rate, bottleneck_bins, n_bins, nonlinear=nonlinear, bias=bias, eps=eps)
        )

    def forward(self, input):
        """
        Args:
            input (batch_size, in_channels, n_bins, n_frames)
        Returns:
            input (batch_size, growth_rate, n_bins, n_frames)
        """
        self.rnn.flatten_parameters()

        x = self.tdc2d(input) # (batch_size, growth_rate, n_bins, n_frames)
        x = self.norm2d(x) # (batch_size, growth_rate, n_bins, n_frames)

        batch_size, growth_rate, n_bins, n_frames = x.size()
        x = x.view(batch_size * growth_rate, n_bins, n_frames)
        x = x.permute(0, 2, 1).contiguous() # (batch_size * growth_rate, n_frames, n_bins)
        x, _ = self.rnn(x) # (batch_size * growth_rate, n_frames, hidden_channels)
        x = x.permute(0, 2, 1) # (batch_size * growth_rate, hidden_channels, n_frames)

        _, hidden_channels, _ = x.size()
        x = x.view(batch_size, growth_rate, hidden_channels, n_frames)
        output = self.tdf2d(x)

        return output

# Control Net
class ControlDenseNet(nn.Module):
    def __init__(self, channels, out_channels, nonlinear='relu', dropout=False, norm=False, eps=EPS):
        """
        Args:
            out_channels <list<int>>: output channels
        """
        super().__init__()

        self.dense_block = ControlStackedDenseBlock(channels, nonlinear=nonlinear, dropout=dropout, norm=norm, eps=eps)

        weights, biases = [], []

        for _channels in out_channels:
            weights.append(nn.Linear(channels[-1], _channels))
            biases.append(nn.Linear(channels[-1], _channels))

        self.fc_weights = nn.ModuleList(weights)
        self.fc_biases = nn.ModuleList(biases)

        self.channels = channels
        self.out_channels = out_channels
        self.nonlinear = nonlinear
        self.dropout = dropout
        self.norm = norm
        self.eps = eps

    def forward(self, input):
        """
        Args:
            input <torch.Tensor>: 
        Returns:
            output_weights <list<torch.Tensor>>: 
            output_biases <list<torch.Tensor>>: 
        """
        out_channels = self.out_channels

        x = self.dense_block(input)

        output_weights, output_biases = [], []

        for idx, _ in enumerate(out_channels):
            x_weights = self.fc_weights[idx](x)
            x_biases = self.fc_biases[idx](x)
            output_weights.append(x_weights)
            output_biases.append(x_biases)

        return output_weights, output_biases

    @classmethod
    def build_from_config(cls, config):
        channels, out_channels = config['channels'], config['out_channels']
        nonlinear = config['nonlinear']
        dropout = config['dropout']
        norm = config['norm']

        model = cls(
            channels, out_channels,
            nonlinear=nonlinear, dropout=dropout, norm=norm
        )

        return model

    def get_config(self):
        config = {
            'channels': self.channels,
            'out_channels': self.out_channels,
            'nonlinear': self.nonlinear,
            'dropout': self.dropout,
            'norm': self.norm,
            'eps': self.eps
        }

        return config

class ControlStackedDenseBlock(nn.Module):
    def __init__(self, channels, nonlinear=False, dropout=False, norm=False, eps=EPS):
        super().__init__()

        n_blocks = len(channels) - 1

        if type(nonlinear) is not list:
            nonlinear = [nonlinear] * n_blocks

        net = []

        for n in range(n_blocks):
            if n == 0: # First layer
                _dropout, _norm = False, False
            else:
                _dropout, _norm = dropout, norm

            net.append(ControlDenseBlock(channels[n], channels[n + 1], nonlinear=nonlinear[n], dropout=_dropout, norm=_norm, eps=eps))

        self.n_blocks = n_blocks
        self.net = nn.Sequential(*net)

    def forward(self, input):
        output = self.net(input)

        return output

class ControlDenseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, nonlinear='relu', dropout=False, norm=False, eps=EPS):
        super().__init__()

        self.nonlinear, self.dropout, self.norm = nonlinear, dropout, norm

        self.linear = nn.Linear(in_channels, out_channels)

        if self.nonlinear:
            self.nonlinear0d = choose_nonlinear(nonlinear)

        if self.dropout:
            self.dropout0d = nn.Dropout(dropout)

        if self.norm:
            self.norm0d = nn.BatchNorm1d(out_channels, eps=eps)

    def forward(self, input):
        """
        Args:
            input (batch_size, in_channels): input tensor
        Returns:
            output (batch_size, out_channels): output tensor
        """
        x = self.linear(input)

        if self.nonlinear: 
            x = self.nonlinear0d(x)

        if self.dropout:
            x = self.dropout0d(x)

        if self.norm:
            x = self.norm0d(x)

        output = x

        return output

class ControlConvNet(nn.Module):
    def __init__(self, channels, out_channels, kernel_size, stride=None, dilated=False, separable=False, nonlinear='relu', dropout=False, norm=False, eps=EPS):
        """
        Args:
            out_channels <list<int>>: output_channels
        """
        super().__init__()

        self.out_channels = out_channels

        self.conv_block = ControlStackedConvBlock(channels, kernel_size=kernel_size, stride=stride, dilated=dilated, separable=separable, nonlinear=nonlinear, dropout=dropout, norm=norm, eps=EPS)

        weights, biases = [], []

        for _channels in out_channels:
            weights.append(nn.Linear(channels[-1], _channels))
            biases.append(nn.Linear(channels[-1], _channels))

        self.fc_weights = nn.ModuleList(weights)
        self.fc_biases = nn.ModuleList(biases)

    def forward(self, input):
        """
        Args:
            input <torch.Tensor>: 
        Returns:
            output_weights <list<torch.Tensor>>: 
            output_biases <list<torch.Tensor>>: 
        """
        batch_size = input.size(0)
        out_channels = self.out_channels

        x = self.conv_block(input)

        assert x.size(-1) == 1, "Invalid tensor shape."

        x = x.view(batch_size, -1)

        output_weights, output_biases = [], []

        for idx, _ in enumerate(out_channels):
            x_weights = self.fc_weights[idx](x)
            x_biases = self.fc_biases[idx](x)
            output_weights.append(x_weights)
            output_biases.append(x_biases)

        return output_weights, output_biases

class ControlStackedConvBlock(nn.Module):
    def __init__(self, channels, kernel_size, stride=None, dilated=False, separable=False, nonlinear=False, dropout=False, norm=False, eps=EPS):
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

        net = []

        for n in range(n_blocks):
            if dilated:
                dilation = 2**(n_blocks - n - 1)
                assert stride[n] == 1, "stride must be 1 when dilated convolution."
            else:
                dilation = 1

            if n == 0: # First layer
                _dropout, _norm = False, False
            else:
                _dropout, _norm = dropout, norm

            net.append(ControlConvBlock(channels[n], channels[n + 1], kernel_size=kernel_size[n], stride=stride[n], dilation=dilation, separable=separable, nonlinear=nonlinear[n], dropout=_dropout, norm=_norm, eps=EPS))

        self.n_blocks = n_blocks
        self.net = nn.Sequential(*net)

    def forward(self, input):
        n_blocks = self.n_blocks

        x = input

        for n in range(n_blocks):
            x = self.net[n](x)

        output = x

        return output

class ControlConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=None, dilation=1, separable=False, nonlinear='relu', dropout=False, norm=False, eps=EPS):
        super().__init__()

        if stride is None:
            stride = kernel_size

        self.kernel_size, self.stride, self.dilation = kernel_size, stride, dilation
        self.nonlinear, self.dropout, self.norm = nonlinear, dropout, norm

        if separable:
            self.conv1d = DepthwiseSeparableConv1d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation)
        else:
            self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation)

        if self.nonlinear:
            self.nonlinear1d = choose_nonlinear(nonlinear)

        if self.dropout:
            self.dropout1d = nn.Dropout(dropout)

        if self.norm:
            self.norm1d = nn.BatchNorm1d(out_channels, eps=eps)

    def forward(self, input):
        """
        Args:
            input (batch_size, in_channels, T_in): input tensor
        Returns:
            output (batch_size, out_channels, T_out): output tensor
        """
        K = self.kernel_size
        S = self.stride
        D = self.dilation

        K = (K - 1) * D + 1

        _, _, T_in = input.size()

        padding_width = K - 1 - (S - (T_in - K) % S) % S
        padding_left = padding_width // 2
        padding_right = padding_width - padding_left

        input = F.pad(input, (padding_left, padding_right))

        x = self.conv1d(input)

        if self.nonlinear: 
            x = self.nonlinear1d(x)

        if self.dropout:
            x = self.dropout1d(x)

        if self.norm:
            x = self.norm1d(x)

        output = x

        return output

def _test_control_net():
    latent_dim = 4
    num_blocks = 6
    dropout, norm = 0.3, True
    batch_size = 2

    channels = [latent_dim, 16, 32, 64]    
    out_channels = [num_blocks] * latent_dim

    input = torch.randn((batch_size, latent_dim), dtype=torch.float)
    model = ControlDenseNet(channels, out_channels, dropout=dropout, norm=norm)
    output_gamma, output_beta = model(input)

    print(model)
    print(input.size())

    for gamma, beta in zip(output_gamma, output_beta):
        print(gamma.size(), beta.size())
    print()

    channels = [1, 16, 32, 64]
    out_channels = [num_blocks] * latent_dim
    kernel_size = [latent_dim, latent_dim, latent_dim]
    stride = [1, 1, latent_dim]

    input = torch.randn((batch_size, 1, latent_dim), dtype=torch.float)
    model = ControlConvNet(channels, out_channels, kernel_size, stride=stride, dropout=dropout, norm=norm)
    output_gamma, output_beta = model(input)

    print(model)
    print(input.size())

    for gamma, beta in zip(output_gamma, output_beta):
        print(gamma.size(), beta.size())

def _test_cunet():
    latent_dim = 4
    n_bins, n_frames = 512, 128

    channels = [1, 8, 16, 32, 64]
    kernel_size = 3
    stride = 2

    dec_nonlinear = ['leaky-relu', 'leaky-relu', 'leaky-relu', 'sigmoid']
    dropout_control = 0.5

    batch_size = 2

    input = torch.randn((batch_size, 1, n_bins, n_frames), dtype=torch.float)

    print('-'*10, "with Complex FiLM and Dense control", '-'*10)
    channels_control = [latent_dim, 4, 8, 16]
    out_channels_control = [1, 1, 1, 1]

    input_latent = torch.randn((batch_size, latent_dim), dtype=torch.float)

    control_net = ControlDenseNet(channels_control, out_channels_control, nonlinear=False, dropout=dropout_control, norm=True)
    unet = UNet2d(channels, kernel_size=kernel_size, stride=stride, dec_nonlinear=dec_nonlinear, conditioning='film')
    model = ConditionedUNet2d(control_net=control_net, unet=unet)
    output = model(input, input_latent)
    print(model)
    print(input.size(), input_latent.size(), output.size())
    print()

    print('-'*10, "with Complex FiLM and Dense control", '-'*10)
    channels_control = [latent_dim, 4, 8, 16]
    out_channels_control = channels[1:]

    input_latent = torch.randn((batch_size, latent_dim), dtype=torch.float)

    control_net = ControlDenseNet(channels_control, out_channels_control, nonlinear=False, dropout=dropout_control, norm=True)
    unet = UNet2d(channels, kernel_size=kernel_size, stride=stride, dec_nonlinear=dec_nonlinear)
    model = ConditionedUNet2d(control_net=control_net, unet=unet, masking=True)
    output = model(input, input_latent)
    print(model)
    print(input.size(), input_latent.size(), output.size())
    print()

    print('-'*10, "with Simple FiLM and Conv control", '-'*10)

    channels_control = [1, 4, 8, 16]
    out_channels_control = [1, 1, 1, 1]
    kernel_size_control = [latent_dim, latent_dim, latent_dim]
    stride_control = [1, 1, latent_dim]

    input_latent = torch.randn((batch_size, 1, latent_dim), dtype=torch.float)

    control_net = ControlConvNet(channels_control, out_channels_control, kernel_size=kernel_size_control, stride=stride_control, dilated=False, separable=False, nonlinear=False, dropout=dropout_control, norm=True)
    unet = UNet2d(channels, kernel_size=kernel_size, stride=stride, dec_nonlinear=dec_nonlinear, conditioning='film')
    model = ConditionedUNet2d(control_net=control_net, unet=unet, masking=True)
    output = model(input, input_latent)
    print(model)
    print(input.size(), input_latent.size(), output.size())
    print()

    print('-'*10, "with Complex FiLM and Conv control", '-'*10)

    channels_control = [1, 4, 8, 16]
    out_channels_control = channels[1:]
    kernel_size_control = [latent_dim, latent_dim, latent_dim]
    stride_control = [1, 1, latent_dim]

    input_latent = torch.randn((batch_size, 1, latent_dim), dtype=torch.float)

    control_net = ControlConvNet(channels_control, out_channels_control, kernel_size=kernel_size_control, stride=stride_control, dilated=False, separable=False, nonlinear=False, dropout=dropout_control, norm=True)
    unet = UNet2d(channels, kernel_size=kernel_size, stride=stride, dec_nonlinear=dec_nonlinear, conditioning='film')
    model = ConditionedUNet2d(control_net=control_net, unet=unet)
    output = model(input, input_latent)
    print(model)
    print(input.size(), input_latent.size(), output.size())

def _test_tfc():
    batch_size = 4
    n_bins, n_frames = 257, 128
    in_channels, growth_rate = 2, 3
    kernel_size = (2, 4)

    input = torch.randn((batch_size, in_channels, n_bins, n_frames), dtype=torch.float)
    model = TFC2d(in_channels, growth_rate=growth_rate, kernel_size=kernel_size)

    output = model(input)
    print(model)
    print(input.size(), output.size())

def _test_tdc():
    batch_size = 4
    n_bins, n_frames = 257, 128
    in_channels, growth_rate = 2, 3
    kernel_size = 3

    input = torch.randn((batch_size, in_channels, n_bins, n_frames), dtype=torch.float)

    model = TDC2d(in_channels, growth_rate=growth_rate, kernel_size=kernel_size)
    output = model(input)

    print(model)
    print(input.size(), output.size())

def _test_tdc_rnn():
    batch_size = 4
    n_bins, n_frames = 257, 128
    in_channels, growth_rate = 2, 7
    kernel_size = 3

    hidden_channels = 16
    bottleneck_bins = 32
    num_layers_tdc, num_layers_rnn = 5, 3

    input = torch.randn((batch_size, in_channels, n_bins, n_frames), dtype=torch.float)

    model = TDCRNN2d(
        in_channels, growth_rate, kernel_size=kernel_size,
        n_bins=n_bins, bottleneck_bins=bottleneck_bins,
        hidden_channels=hidden_channels,
        num_layers_tdc=num_layers_tdc, num_layers_rnn=num_layers_rnn
    )
    output = model(input)

    print(model)
    print(input.size(), output.size())

if __name__ == '__main__':
    print("="*10, "Control Net", "="*10)
    _test_control_net()
    print()

    print("="*10, "Conditioned-U-Net", "="*10)
    _test_cunet()
    print()

    print("="*10, "Time-frequency Convolution", "="*10)
    _test_tfc()
    print()

    print("="*10, "Time-distributed Convolution", "="*10)
    _test_tdc()

    print("="*10, "Time-distributed Convolution with RNN", "="*10)
    _test_tdc_rnn()