import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from conv import DepthwiseSeparableConv1d, DepthwiseSeparableConv2d, DepthwiseSeparableConvTranspose2d
from models.film import FiLM2d

"""
Conditioned-U-Net: Introducing a Control Mechanism in the U-Net for multiple source separations
"""

class ConditionedUNetBase(nn.Module):
    def __init__(self):
        super().__init__()
        
    def _get_num_parameters(self):
        num_parameters = 0
        
        for p in self.parameters():
            if p.requires_grad:
                num_parameters += p.numel()
                
        return num_parameters

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
        
        self.num_parameters = self._get_num_parameters()
        
    def forward(self, input, latent):
        gamma, beta = self.control_net(latent)
        x = self.backbone(input, gamma, beta)

        _, _, H_in, W_in = input.size()
        _, _, H, W = x.size()
        padding_height = H - H_in
        padding_width = W - W_in
        padding_top = padding_height//2
        padding_bottom = padding_height - padding_top
        padding_left = padding_width//2
        padding_right = padding_width - padding_left

        x = F.pad(x, (-padding_left, -padding_right, -padding_top, -padding_bottom))

        if self.masking:
            output = x * input
        else:
            output = x
        
        return output

class UNet2d(ConditionedUNetBase):
    def __init__(
            self,
            channels,
            kernel_size, stride=None,
            dilated=False,
            nonlinear_enc='leaky-relu', nonlinear_dec='leaky-relu',
            out_channels=None
        ):
        """
        Args:
            channels <list<int>>:
        """
        super().__init__()
        
        channels_enc = channels
        
        if out_channels is None:
            channels_dec = channels[::-1]
        else:
            channels_dec = channels[:0:-1] + [out_channels]
            
        _channels_dec = []
        
        for idx, out_channel in enumerate(channels_dec):
            if idx == 0:
                _channels_dec.append(out_channel)
            else:
                _channels_dec.append(2 * out_channel)
                
        channels_dec = _channels_dec

        self.encoder = Encoder2d(channels_enc, kernel_size=kernel_size, stride=stride, dilated=dilated, nonlinear=nonlinear_enc)
        self.bottleneck = nn.Conv2d(channels[-1], channels[-1], kernel_size=(1,1), stride=(1,1))
        self.decoder = Decoder2d(channels_dec, kernel_size=kernel_size, stride=stride, dilated=dilated, nonlinear=nonlinear_dec)
        
    def forward(self, input, gamma, beta):
        x, skip = self.encoder(input, gamma, beta)
        x = self.bottleneck(x)
        output = self.decoder(x, skip[::-1])
        
        return output

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
        
        self.channels = channels
        self.n_blocks = n_blocks
        
        net = []
        
        for n in range(n_blocks):
            if dilated:
                dilation = 2**n
                assert stride[n] == 1, "stride must be 1 when dilated convolution."
            else:
                dilation = 1
            net.append(EncoderBlock2d(channels[n], channels[n+1], kernel_size=kernel_size[n], stride=stride[n], dilation=dilation, separable=separable, nonlinear=nonlinear[n]))
        
        self.net = nn.Sequential(*net)
        
    def forward(self, input, gamma, beta):
        n_blocks = self.n_blocks
        
        x = input
        skip = []

        # print(x.size(), gamma.size(), beta.size(), self.channels)
        
        for n in range(n_blocks):
            x = self.net[n](x, gamma[n], beta[n])
            skip.append(x)
        
        return x, skip

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
            net.append(DecoderBlock2d(channels[n], channels[n+1]//2, kernel_size=kernel_size[n], stride=stride[n], dilation=dilation, separable=separable, nonlinear=nonlinear[n]))
            # channels[n+1]//2: because of skip connection
        
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
        
        self.batch_norm2d = nn.BatchNorm2d(out_channels)
        self.film = FiLM2d()
        
        if nonlinear == 'relu':
            self.nonlinear = nn.ReLU()
        elif nonlinear == 'leaky-relu':
            self.nonlinear = nn.LeakyReLU()
        else:
            raise NotImplementedError()
            
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
        x = self.batch_norm2d(x)
        x = self.film(x, gamma, beta)
        output = self.nonlinear(x)
        
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
        self.batch_norm2d = nn.BatchNorm2d(out_channels)
        
        if nonlinear == 'relu':
            self.nonlinear = nn.ReLU()
        elif nonlinear == 'sigmoid':
            self.nonlinear = nn.Sigmoid()
        elif nonlinear == 'leaky-relu':
            self.nonlinear = nn.LeakyReLU()
        else:
            raise NotImplementedError()
            
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
        x = self.batch_norm2d(x)
        output = self.nonlinear(x)
        
        return output

class ControlDenseNet(nn.Module):
    def __init__(self, channels, out_channels, nonlinear='relu', dropout=False, norm=False):
        """
        Args:
            out_channels <list<int>>: output_channels
        """
        super().__init__()

        self.out_channels = out_channels

        self.dense_block = ControlStackedDenseBlock(channels, nonlinear=nonlinear, dropout=dropout, norm=norm)

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
        out_channels = self.out_channels

        x = self.dense_block(input)

        output_weights, output_biases = [], []

        for idx, _ in enumerate(out_channels):
            x_weights = self.fc_weights[idx](x)
            x_biases = self.fc_biases[idx](x)
            output_weights.append(x_weights)
            output_biases.append(x_biases)

        return output_weights, output_biases

class ControlStackedDenseBlock(nn.Module):
    def __init__(self, channels, nonlinear=False, dropout=False, norm=False):
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

            net.append(ControlDenseBlock(channels[n], channels[n + 1], nonlinear=nonlinear[n], dropout=_dropout, norm=_norm))

        self.n_blocks = n_blocks
        self.net = nn.Sequential(*net)
        
    def forward(self, input):
        output = self.net(input)

        return output

class ControlDenseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, nonlinear='relu', dropout=False, norm=False):
        super().__init__()

        self.nonlinear, self.dropout, self.norm = nonlinear, dropout, norm

        self.linear = nn.Linear(in_channels, out_channels)

        if self.nonlinear:
            if self.nonlinear == 'relu':
                self.nonlinear0d = nn.ReLU()
            elif self.nonlinear == 'leaky-relu':
                self.nonlinear0d = nn.LeakyReLU()
            else:
                raise ValueError("Not support nonlinear {}".format(self.nonlinear))

        if self.dropout:
            self.dropout0d = nn.Dropout(dropout)

        if self.norm:
            self.batch_norm0d = nn.BatchNorm1d(out_channels)
    
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
            x = self.batch_norm0d(x)
        
        output = x

        return output

class ControlConvNet(nn.Module):
    def __init__(self, channels, out_channels, kernel_size, stride=None, dilated=False, separable=False, nonlinear='relu', dropout=False, norm=False):
        """
        Args:
            out_channels <list<int>>: output_channels
        """
        super().__init__()

        self.out_channels = out_channels

        self.conv_block = ControlStackedConvBlock(channels, kernel_size=kernel_size, stride=stride, dilated=dilated, separable=separable, nonlinear=nonlinear, dropout=dropout, norm=norm)

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
    def __init__(self, channels, kernel_size, stride=None, dilated=False, separable=False, nonlinear=False, dropout=False, norm=False):
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

            net.append(ControlConvBlock(channels[n], channels[n + 1], kernel_size=kernel_size[n], stride=stride[n], dilation=dilation, separable=separable, nonlinear=nonlinear[n], dropout=_dropout, norm=_norm))

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
    def __init__(self, in_channels, out_channels, kernel_size, stride=None, dilation=1, separable=False, nonlinear='relu', dropout=False, norm=False):
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
            if self.nonlinear == 'relu':
                self.nonlinear1d = nn.ReLU()
            elif self.nonlinear == 'leaky-relu':
                self.nonlinear1d = nn.LeakyReLU()
            else:
                raise ValueError("Not support nonlinear {}".format(self.nonlinear))

        if self.dropout:
            self.dropout1d = nn.Dropout(dropout)

        if self.norm:
            self.batch_norm1d = nn.BatchNorm1d(out_channels)
    
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
            x = self.batch_norm1d(x)
        
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
    
    nonlinear_dec = ['leaky-relu', 'leaky-relu', 'leaky-relu', 'sigmoid']
    dropout_control = 0.5

    batch_size = 2

    channels_control = [latent_dim, 4, 8, 16]

    input = torch.randn((batch_size, 1, n_bins, n_frames), dtype=torch.float)

    input_latent = torch.randn((batch_size, latent_dim), dtype=torch.float)
    control_net = ControlDenseNet(channels_control, channels[1:], nonlinear=False, dropout=dropout_control, norm=True)
    unet = UNet2d(channels, kernel_size=kernel_size, stride=stride, nonlinear_dec=nonlinear_dec)
    model = ConditionedUNet2d(control_net=control_net, unet=unet, masking=True)
    output = model(input, input_latent)
    print(model)
    print(input.size(), input_latent.size(), output.size())
    print()

    channels_control = [1, 4, 8, 16]
    kernel_size_control = [latent_dim, latent_dim, latent_dim]
    stride_control = [1, 1, latent_dim]

    input_latent = torch.randn((batch_size, 1, latent_dim), dtype=torch.float)
    control_net = ControlConvNet(channels_control, channels[1:], kernel_size=kernel_size_control, stride=stride_control, dilated=False, separable=False, nonlinear=False, dropout=dropout_control, norm=True)
    unet = UNet2d(channels, kernel_size=kernel_size, stride=stride, nonlinear_dec=nonlinear_dec)
    model = ConditionedUNet2d(control_net=control_net, unet=unet, masking=True)
    output = model(input, input_latent)
    print(model)
    print(input.size(), input_latent.size(), output.size())

if __name__ == '__main__':
    print("="*10, "Control Net", "="*10)
    _test_control_net()
    print()

    print("="*10, "Conditioned-U-Net", "="*10)
    _test_cunet()