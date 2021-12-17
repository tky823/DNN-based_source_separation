import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from utils.model import choose_nonlinear
from conv import DepthwiseSeparableConv1d, DepthwiseSeparableConvTranspose1d, DepthwiseSeparableConv2d, DepthwiseSeparableConvTranspose2d

EPS = 1e-12

class UNetBase(nn.Module):
    def __init__(self, eps=EPS):
        super().__init__()

        self.eps = eps

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
            'kernel_size': self.kernel_size, 'stride': self.stride,
            'dilated': self.dilated,
            'enc_nonlinear': self.enc_nonlinear, 'dec_nonlinear': self.dec_nonlinear,
            'out_channels': self.out_channels,
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

class UNet1d(UNetBase):
    def __init__(self, channels, kernel_size, stride=None, dilated=False, enc_nonlinear='relu', dec_nonlinear='relu', out_channels=None, eps=EPS):
        """
        Args:
            channels <list<int>>
            kernel_size <int>
            stride <int>
            dilated <bool>
            enc_nonlinear <str>
            dec_nonlinear <str>
            out_channels <int>
            eps <float>
        """
        super().__init__(eps=eps)

        enc_channels = channels
        dec_channels = channels[::-1] if out_channels is None else channels[:0:-1] + [out_channels]
        dec_channels = [out_channel if idx == 0 else 2 * out_channel for idx, out_channel in enumerate(dec_channels)]

        self.channels = channels
        self.kernel_size, self.stride, self.dilated = kernel_size, stride, dilated
        self.enc_nonlinear, self.dec_nonlinear = enc_nonlinear, dec_nonlinear
        self.out_channels = out_channels

        self.encoder = Encoder1d(enc_channels, kernel_size=kernel_size, stride=stride, dilated=dilated, nonlinear=enc_nonlinear, eps=eps)
        self.bottleneck = nn.Conv1d(channels[-1], channels[-1], kernel_size=1, stride=1)
        self.decoder = Decoder1d(dec_channels, kernel_size=kernel_size, stride=stride, dilated=dilated, nonlinear=dec_nonlinear, eps=eps)

    def forward(self, input):
        """
        Args:
            input: (batch_size, in_channels, T)
        Returns:
            output: (batch_size, out_channels, T)
        """
        x, skip = self.encoder(input)
        x = self.bottleneck(x)
        output = self.decoder(x, skip[::-1])

        T_in, T_out = input.size(-1), output.size(-1)
        P = T_out - T_in
        P_left = P // 2
        P_right = P - P_left

        output = F.pad(output, (-P_left, -P_right))

        return output

class UNet2d(UNetBase):
    def __init__(self, channels, kernel_size, stride=None, dilated=False, enc_nonlinear='relu', dec_nonlinear='relu', out_channels=None, eps=EPS):
        """
        Args:
            channels <list<int>>
            kernel_size <int>
            stride <int>
            dilated <bool>
            enc_nonlinear <str>
            dec_nonlinear <str>
            out_channels <int>
            eps <float>
        """
        super().__init__(eps=eps)
        
        enc_channels = channels
        dec_channels = channels[::-1] if out_channels is None else channels[:0:-1] + [out_channels]
        dec_channels = [out_channel if idx == 0 else 2 * out_channel for idx, out_channel in enumerate(dec_channels)]
        
        self.channels = channels
        self.kernel_size, self.stride, self.dilated = kernel_size, stride, dilated
        self.enc_nonlinear, self.dec_nonlinear = enc_nonlinear, dec_nonlinear
        self.out_channels = out_channels

        self.encoder = Encoder2d(enc_channels, kernel_size=kernel_size, stride=stride, dilated=dilated, nonlinear=enc_nonlinear, eps=eps)
        self.bottleneck = nn.Conv2d(channels[-1], channels[-1], kernel_size=(1,1), stride=(1,1))
        self.decoder = Decoder2d(dec_channels, kernel_size=kernel_size, stride=stride, dilated=dilated, nonlinear=dec_nonlinear, eps=eps)

    def forward(self, input):
        """
        Args:
            input: (batch_size, in_channels, H, W)
        Returns:
            output: (batch_size, out_channels, H, W)
        """
        x, skip = self.encoder(input)
        x = self.bottleneck(x)
        output = self.decoder(x, skip[::-1])

        (H_in, W_in), (H_out, W_out) = input.size()[-2:], output.size()[-2:]
        Ph, Pw = H_out - H_in, W_out - W_in
        Ph_top, Pw_left = Ph // 2, Pw // 2
        Ph_bottom, Pw_right = Ph - Ph_top, Pw - Pw_left

        output = F.pad(output, (-Pw_left, -Pw_right, -Ph_top, -Ph_bottom))

        return output

class EnsembleUNet1d(UNetBase):
    def __init__(self, channels, kernel_size, stride=None, dilated=False, enc_nonlinear='relu', dec_nonlinear='relu', out_channels=None, eps=EPS):
        """
        Args:
            channels <list<int>>
            kernel_size <int>
            stride <int>
            dilated <bool>
            enc_nonlinear <str>
            dec_nonlinear <str>
            out_channels <int>
            eps <float>
        """
        super().__init__()

        num_ensembles = len(channels) - 1

        dec_channels = channels[::-1] if out_channels is None else channels[:0:-1] + [out_channels]
        dec_channels = [out_channel if idx == 0 else 2 * out_channel for idx, out_channel in enumerate(dec_channels)]

        kernel_size = kernel_size if type(kernel_size) is list else [kernel_size] * num_ensembles
        stride = stride if type(stride) is list else [stride] * num_ensembles
        enc_nonlinear = enc_nonlinear if type(enc_nonlinear) is list else [enc_nonlinear] * num_ensembles
        dec_nonlinear = dec_nonlinear if type(dec_nonlinear) is list else [dec_nonlinear] * num_ensembles

        self.channels = channels
        self.kernel_size, self.stride, self.dilated = kernel_size, stride, dilated
        self.enc_nonlinear, self.dec_nonlinear = enc_nonlinear, dec_nonlinear
        self.out_channels = out_channels
        self.num_ensembles = num_ensembles

        encoder, decoders = [], []

        for ensemble_idx in range(self.num_ensembles):
            if dilated:
                dilation = 2**ensemble_idx
                assert stride[ensemble_idx] == 1, "stride must be 1 when dilated convolution."
            else:
                dilation = 1

            _channels = channels[:ensemble_idx+2]
            _dec_channels = _channels[::-1] if out_channels is None else _channels[:0:-1] + [out_channels]
            _dec_channels = [channel if idx == 0 else 2 * channel for idx, channel in enumerate(_dec_channels)]
            _kernel_size, _stride = kernel_size[:ensemble_idx+1], stride[:ensemble_idx+1]
            _enc_nonlinear, _dec_nonlinear = enc_nonlinear[ensemble_idx], dec_nonlinear[:ensemble_idx+1]

            encoder_block = EncoderBlock1d(_channels[-2], _channels[-1], kernel_size=_kernel_size[-1], stride=_stride[-1], dilation=dilation, nonlinear=_enc_nonlinear, eps=eps)
            decoder = BottleneckDecoder1d(_dec_channels, _kernel_size, stride=_stride, dilated=dilated, nonlinear=_dec_nonlinear, eps=eps)
            encoder.append(encoder_block)
            decoders.append(decoder)

        self.encoder, self.decoders = nn.ModuleList(encoder), nn.ModuleList(decoders)

    def forward(self, input, stack_dim=1, return_all_layers=True):
        """
        Args:
            input: (batch_size, in_channels, T)
        Returns:
            output: (batch_size, out_channels, T)
        """
        outputs = []

        x = input
        skip = []

        for ensemble_idx in range(self.num_ensembles):
            x = self.encoder[ensemble_idx](x)
            skip.append(x)
            output = self.decoders[ensemble_idx](x, skip[::-1], return_all_layers=False)
            outputs.append(output)

        if return_all_layers:
            outputs = torch.stack(outputs, dim=stack_dim)
        else:
            outputs = outputs[-1]

        return outputs

class EnsembleUNet2d(UNetBase):
    def __init__(self, channels, kernel_size, stride=None, dilated=False, enc_nonlinear='relu', dec_nonlinear='relu', out_channels=None, eps=EPS):
        """
        Args:
            channels <list<int>>
            kernel_size <int>
            stride <int>
            dilated <bool>
            enc_nonlinear <str>
            dec_nonlinear <str>
            out_channels <int>
            eps <float>
        """
        super().__init__()

        num_ensembles = len(channels) - 1

        dec_channels = channels[::-1] if out_channels is None else channels[:0:-1] + [out_channels]
        dec_channels = [out_channel if idx == 0 else 2 * out_channel for idx, out_channel in enumerate(dec_channels)]

        kernel_size = kernel_size if type(kernel_size) is list else [kernel_size] * num_ensembles
        stride = stride if type(stride) is list else [stride] * num_ensembles
        enc_nonlinear = enc_nonlinear if type(enc_nonlinear) is list else [enc_nonlinear] * num_ensembles
        dec_nonlinear = dec_nonlinear if type(dec_nonlinear) is list else [dec_nonlinear] * num_ensembles

        self.channels = channels
        self.kernel_size, self.stride, self.dilated = kernel_size, stride, dilated
        self.enc_nonlinear, self.dec_nonlinear = enc_nonlinear, dec_nonlinear
        self.out_channels = out_channels
        self.num_ensembles = num_ensembles

        encoder, decoders = [], []

        for ensemble_idx in range(self.num_ensembles):
            if dilated:
                dilation = 2**ensemble_idx
                assert stride[ensemble_idx] == 1, "stride must be 1 when dilated convolution."
            else:
                dilation = 1

            _channels = channels[:ensemble_idx+2]
            _dec_channels = _channels[::-1] if out_channels is None else _channels[:0:-1] + [out_channels]
            _dec_channels = [channel if idx == 0 else 2 * channel for idx, channel in enumerate(_dec_channels)]
            _kernel_size, _stride = kernel_size[:ensemble_idx+1], stride[:ensemble_idx+1]
            _enc_nonlinear, _dec_nonlinear = enc_nonlinear[ensemble_idx], dec_nonlinear[:ensemble_idx+1]

            encoder_block = EncoderBlock2d(_channels[-2], _channels[-1], kernel_size=_kernel_size[-1], stride=_stride[-1], dilation=dilation, nonlinear=_enc_nonlinear, eps=eps)
            decoder = BottleneckDecoder2d(_dec_channels, _kernel_size, stride=_stride, dilated=dilated, nonlinear=_dec_nonlinear, eps=eps)
            encoder.append(encoder_block)
            decoders.append(decoder)

        self.encoder, self.decoders = nn.ModuleList(encoder), nn.ModuleList(decoders)

    def forward(self, input, stack_dim=1, return_all_layers=True):
        """
        Args:
            input: (batch_size, in_channels, H, W)
        Returns:
            output: (batch_size, out_channels, H, W)
        """
        outputs = []

        x = input
        skip = []

        for ensemble_idx in range(self.num_ensembles):
            x = self.encoder[ensemble_idx](x)
            skip.append(x)
            output = self.decoders[ensemble_idx](x, skip[::-1], return_all_layers=False)
            outputs.append(output)

        if return_all_layers:
            outputs = torch.stack(outputs, dim=stack_dim)
        else:
            outputs = outputs[-1]

        return outputs

"""
    Encoder
"""
class Encoder1d(nn.Module):
    def __init__(self, channels, kernel_size, stride=None, dilated=False, separable=False, nonlinear='relu', eps=EPS):
        """
        Args:
            channels <list<int>>
            kernel_size <int> or <list<int>>
            stride <int> or <list<int>>
            dilated <bool>
            nonlinear <str> or <list<str>>
        """
        super().__init__()

        num_blocks = len(channels) - 1

        if type(kernel_size) is not list:
            kernel_size = [kernel_size] * num_blocks
        if stride is None:
            stride = kernel_size
        elif type(stride) is not list:
            stride = [stride] * num_blocks
        if type(nonlinear) is not list:
            nonlinear = [nonlinear] * num_blocks

        self.num_blocks = num_blocks

        net = []

        for n in range(num_blocks):
            if dilated:
                dilation = 2**n
                assert stride[n] == 1, "stride must be 1 when dilated convolution."
            else:
                dilation = 1

            net.append(EncoderBlock1d(channels[n], channels[n+1], kernel_size=kernel_size[n], stride=stride[n], dilation=dilation, separable=separable, nonlinear=nonlinear[n], eps=eps))

        self.net = nn.Sequential(*net)

    def forward(self, input):
        num_blocks = self.num_blocks

        x = input
        skip = []

        for n in range(num_blocks):
            x = self.net[n](x)
            skip.append(x)

        return x, skip

class Encoder2d(nn.Module):
    def __init__(self, channels, kernel_size, stride=None, dilated=False, separable=False, nonlinear='relu', eps=EPS):
        """
        Args:
            channels <list<int>>
            kernel_size <tuple<int,int>> or <list<tuple<int,int>>>
            stride <tuple<int,int>> or <list<tuple<int,int>>>
            dilated <bool>
            nonlinear <str> or <list<str>>
        """
        super().__init__()

        num_blocks = len(channels) - 1

        if type(kernel_size) is not list:
            kernel_size = _pair(kernel_size)
            kernel_size = [kernel_size] * num_blocks
        if stride is None:
            stride = kernel_size
        elif type(stride) is not list:
            stride = [stride] * num_blocks
        if type(nonlinear) is not list:
            nonlinear = [nonlinear] * num_blocks

        self.num_blocks = num_blocks

        net = []

        for n in range(num_blocks):
            if dilated:
                dilation = 2**n
                assert stride[n] == 1, "stride must be 1 when dilated convolution."
            else:
                dilation = 1

            net.append(EncoderBlock2d(channels[n], channels[n + 1], kernel_size=kernel_size[n], stride=stride[n], dilation=dilation, separable=separable, nonlinear=nonlinear[n], eps=eps))

        self.net = nn.Sequential(*net)

    def forward(self, input):
        num_blocks = self.num_blocks

        x = input
        skip = []

        for n in range(num_blocks):
            x = self.net[n](x)
            skip.append(x)

        return x, skip

"""
    Decoder
"""
class Decoder1d(nn.Module):
    def __init__(self, channels, kernel_size, stride=None, dilated=False, separable=False, nonlinear='relu', eps=EPS):
        """
        Args:
            channels <list<int>>
            kernel_size <int> or <list<int>>
            stride <int> or <list<int>>
            dilated <bool>
            nonlinear <str> or <list<str>>
        """
        super().__init__()

        num_blocks = len(channels) - 1

        if type(kernel_size) is not list:
            kernel_size = [kernel_size] * num_blocks
        if stride is None:
            stride = kernel_size
        elif type(stride) is not list:
            stride = [stride] * num_blocks
        if type(nonlinear) is not list:
            nonlinear = [nonlinear] * num_blocks

        self.num_blocks = num_blocks

        net = []

        for n in range(num_blocks):
            if dilated:
                dilation = 2**(num_blocks - n - 1)
                assert stride[n] == 1, "stride must be 1 when dilated convolution."
            else:
                dilation = 1
            net.append(DecoderBlock1d(channels[n], channels[n + 1] // 2, kernel_size=kernel_size[n], stride=stride[n], dilation=dilation, separable=separable, nonlinear=nonlinear[n], eps=eps))
            # channels[n + 1] // 2: because of skip connection

        self.net = nn.Sequential(*net)

    def forward(self, input, skip, return_all_layers=False):
        """
        Args:
            input (batch_size, C, T)
            skip <list<torch.Tensor>>
            return_all_layers <bool>
        Returns:
            outputs:
                (batch_size, C_out, T_out)
                <list<torch.Tensor>> if return_all_layers=True
                <torch.Tensor> with shape of (batch_size, C_out, T_out) otherwise
        """
        num_blocks = self.num_blocks

        outputs = []
        x = input

        for n in range(num_blocks):
            if n == 0:
                x = self.net[n](x)
            else:
                x = self.net[n](x, skip[n])

            outputs.append(x)

        if return_all_layers:
            outputs = outputs
        else:
            outputs = outputs[-1]

        return outputs

class Decoder2d(nn.Module):
    def __init__(self, channels, kernel_size, stride=None, dilated=False, separable=False, nonlinear='relu', eps=EPS):
        """
        Args:
            channels <list<int>>
            kernel_size <tuple<int,int>> or <list<tuple<int,int>>>
            stride <tuple<int,int>> or <list<tuple<int,int>>>
            dilated <bool>
            nonlinear <str> or <list<str>>
        """
        super().__init__()

        num_blocks = len(channels) - 1

        if type(kernel_size) is not list:
            kernel_size = [kernel_size] * num_blocks
        if stride is None:
            stride = kernel_size
        elif type(stride) is not list:
            stride = [stride] * num_blocks
        if type(nonlinear) is not list:
            nonlinear = [nonlinear] * num_blocks

        self.num_blocks = num_blocks

        net = []

        for n in range(num_blocks):
            if dilated:
                dilation = 2**(num_blocks - n - 1)
                assert stride[n] == 1, "stride must be 1 when dilated convolution."
            else:
                dilation = 1

            net.append(DecoderBlock2d(channels[n], channels[n + 1] // 2, kernel_size=kernel_size[n], stride=stride[n], dilation=dilation, separable=separable, nonlinear=nonlinear[n], eps=eps))
            # channels[n + 1] // 2: because of skip connection

        self.net = nn.Sequential(*net)

    def forward(self, input, skip, return_all_layers=False):
        """
        Args:
            input (batch_size, C, H, W)
            skip <list<torch.Tensor>>
            return_all_layers <bool>
        Returns:
            outputs:
                <list<torch.Tensor>> if return_all_layers=True
                <torch.Tensor> with shape of (batch_size, C_out, H_out, W_out) otherwise
        """
        num_blocks = self.num_blocks

        outputs = []
        x = input

        for n in range(num_blocks):
            if n == 0:
                x = self.net[n](x)
            else:
                x = self.net[n](x, skip[n])

            outputs.append(x)

        if return_all_layers:
            outputs = outputs
        else:
            outputs = outputs[-1]

        return outputs

class BottleneckDecoder1d(nn.Module):
    def __init__(self, channels, kernel_size, stride=None, dilated=False, nonlinear='relu', eps=EPS):
        super().__init__()

        bottleneck_channels = channels[0]
        self.bottleneck_conv1d = nn.Conv1d(bottleneck_channels, bottleneck_channels, kernel_size=1, stride=1)
        self.decoder = Decoder1d(channels, kernel_size=kernel_size, stride=stride, dilated=dilated, nonlinear=nonlinear, eps=eps)

    def forward(self, input, skip, return_all_layers=False):
        """
        Args:
            input <torch.Tensor>: (batch_size, C_in, T_in)
            skip <list<torch.Tensor>>
        Returns:
            output:
                <list<torch.Tensor>> if return_all_layers=True
                <torch.Tensor> with shape of (batch_size, C_out, T_out) otherwise
        """
        x = self.bottleneck_conv1d(input)
        output = self.decoder(x, skip, return_all_layers=return_all_layers)

        return output

class BottleneckDecoder2d(nn.Module):
    def __init__(self, channels, kernel_size, stride=None, dilated=False, nonlinear='relu', eps=EPS):
        super().__init__()

        bottleneck_channels = channels[0]
        self.bottleneck_conv2d = nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=1, stride=1)
        self.decoder = Decoder2d(channels, kernel_size=kernel_size, stride=stride, dilated=dilated, nonlinear=nonlinear, eps=eps)

    def forward(self, input, skip, return_all_layers=False):
        """
        Args:
            input <torch.Tensor>: (batch_size, C_in, H_in, W_in)
            skip <list<torch.Tensor>>
        Returns:
            output:
                <list<torch.Tensor>> if return_all_layers=True
                <torch.Tensor> with shape of (batch_size, C_out, H_out, W_out) otherwise
        """
        x = self.bottleneck_conv2d(input)
        output = self.decoder(x, skip, return_all_layers=return_all_layers)

        return output

"""
    Encoder Block
"""
class EncoderBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=None, dilation=1, separable=False, nonlinear='relu', eps=EPS):
        super().__init__()

        if stride is None:
            stride = kernel_size

        self.kernel_size, self.stride, self.dilation = kernel_size, stride, dilation

        if separable:
            self.conv1d = DepthwiseSeparableConv1d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation)
        else:
            self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation)

        self.norm1d = nn.BatchNorm1d(out_channels, eps=eps)

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
        K, S, D = self.kernel_size, self.stride, self.dilation
        K = (K - 1) * D + 1

        T = input.size(-1)
        P = K - 1 - (S - (T - K) % S) % S
        P_left = P // 2
        P_right = P - P_left

        input = F.pad(input, (P_left, P_right))

        x = self.conv1d(input)
        x = self.norm1d(x)
        output = self.nonlinear(x)

        return output

class EncoderBlock2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=None, dilation=1, separable=False, nonlinear='relu', eps=EPS):
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

        self.norm2d = nn.BatchNorm2d(out_channels, eps=eps)

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
        (Kh, Kw), (Sh, Sw) = self.kernel_size, self.stride
        Dh, Dw = self.dilation
        Kh, Kw = (Kh - 1) * Dh + 1, (Kw - 1) * Dw + 1

        H, W = input.size()[-2:]
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
    def __init__(self, in_channels, out_channels, kernel_size, stride=None, dilation=1, separable=False, nonlinear='relu', eps=EPS):
        super().__init__()

        if stride is None:
            stride = kernel_size

        self.kernel_size, self.stride, self.dilation = kernel_size, stride, dilation

        if separable:
            self.deconv1d = DepthwiseSeparableConvTranspose1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation)
        else:
            self.deconv1d = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation)

        self.norm1d = nn.BatchNorm1d(out_channels, eps=eps)

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
            input = self.concat_skip(input, skip)

        x = self.deconv1d(input)
        x = F.pad(x, (-P_left, -P_right))
        x = self.norm1d(x)
        output = self.nonlinear(x)

        return output

    def concat_skip(self, input, skip):
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
    def __init__(self, in_channels, out_channels, kernel_size, stride=None, dilation=1, separable=False, nonlinear='relu', eps=EPS):
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

        self.norm2d = nn.BatchNorm2d(out_channels, eps=eps)

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
            input = self.concat_skip(input, skip)

        x = self.deconv2d(input)
        x = F.pad(x, (-Pw_left, -Pw_right, -Ph_top, -Ph_bottom))
        x = self.norm2d(x)
        output = self.nonlinear(x)

        return output

    def concat_skip(self, input, skip):
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

    package = unet2d.get_config()
    model_path = "unet.pth"
    torch.save(package, model_path)
    _ = UNet2d.build_model(model_path)

if __name__ == '__main__':
    _test_unet()
