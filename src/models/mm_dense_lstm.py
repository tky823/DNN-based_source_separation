import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from utils.utils_mm_dense_lstm import choose_layer_norm, choose_nonlinear
from models.m_densenet import ConvBlock2d, DenseBlock

EPS = 1e-12

class MMDenseLSTM(nn.Module):
    def __init__(self, combination='parallel'):
        super().__init__()

class Encoder(nn.Module):
    def __init__(self, in_channels, growth_rate, kernel_size, down_scale=(2,2), num_d2blocks=None, dilated=False, norm=True, nonlinear='relu', depth=None, eps=EPS):
        """
        Args:
            in_channels <int>: 
            growth_rate <list<int>>:
            kernel_size <tuple<int>> or <int>:
            num_d2blocks <list<int>> or <int>:
            dilated <list<bool>> or <bool>:
            norm <list<bool>> or <bool>:
            nonlinear <list<str>> or <str>:
        """
        super().__init__()

        if type(growth_rate) is list:
            num_dense_blocks = len(growth_rate)
        else:
            # TODO: implement
            raise ValueError("`growth_rate` must be list.")

        if type(dilated) is bool:
            dilated = [dilated] * num_dense_blocks
        elif type(dilated) is list:
            assert num_dense_blocks == len(dilated), "Invalid length of `dilated`"
        else:
            raise ValueError("Invalid type of `dilated`.")

        if type(norm) is bool:
            norm = [norm] * num_dense_blocks
        elif type(norm) is list:
            assert num_dense_blocks == len(norm), "Invalid length of `norm`"
        else:
            raise ValueError("Invalid type of `norm`.")

        if type(nonlinear) is str:
            nonlinear = [nonlinear] * num_dense_blocks
        elif type(nonlinear) is list:
            assert num_dense_blocks == len(nonlinear), "Invalid length of `nonlinear`"
        else:
            raise ValueError("Invalid type of `nonlinear`.")
        
        if depth is None:
            depth = [None] * num_dense_blocks
        elif type(depth) is int:
            depth = [depth] * num_dense_blocks
        elif type(depth) is list:
            assert num_dense_blocks == len(depth), "Invalid length of `depth`"
        else:
            raise ValueError("Invalid type of `depth`.")

        num_dense_blocks = len(growth_rate)
        net = []

        _in_channels = in_channels

        for idx in range(num_dense_blocks):
            downsample_block = DownSampleDenseBlock(_in_channels, growth_rate[idx], kernel_size=kernel_size, down_scale=down_scale, num_blocks=num_d2blocks[idx], dilated=dilated[idx], norm=norm[idx], nonlinear=nonlinear[idx], depth=depth[idx], eps=eps)
            net.append(downsample_block)
            _in_channels = downsample_block.out_channels
        
        self.net = nn.Sequential(*net)

        self.num_dense_blocks = num_dense_blocks
    
    def forward(self, input):
        num_dense_blocks = self.num_dense_blocks

        x = input
        skip = []

        for idx in range(num_dense_blocks):
            x, x_skip = self.net[idx](x)
            skip.append(x_skip)
        
        output = x

        return output, skip

class DownSampleDenseLSTMBlock(nn.Module):
    """
    DenseBlock + down sample
    """
    def __init__(self, in_channels, growth_rate, kernel_size=(3,3), down_scale=(2,2), dilated=False, norm=True, nonlinear='relu', depth=None, position_rnn='parallel', eps=EPS):
        super().__init__()

        self.down_scale = _pair(down_scale)

        self.dense_block = DenseLSTMBlock(in_channels, growth_rate, kernel_size, dilated=dilated, norm=norm, nonlinear=nonlinear, depth=depth, position_rnn=position_rnn, eps=eps)
        self.downsample2d = nn.AvgPool2d(kernel_size=self.down_scale, stride=self.down_scale)
    
    def forward(self, input):
        """
        Args:
            input (batch_size, in_channels, H, W)
        Returns:
            output:
                (batch_size, growth_rate[-1], H_down, W_down) if type(growth_rate) is list<int>
                or (batch_size, growth_rate, H_down, W_down) if type(growth_rate) is int
                where H_down = H // down_scale[0] and W_down = W // down_scale[1]
            skip:
                (batch_size, growth_rate[-1], H, W) if type(growth_rate) is list<int>
                or (batch_size, growth_rate, H, W) if type(growth_rate) is int
        """
        _, _, n_bins, n_frames = input.size()

        Kh, Kw = self.down_scale
        Ph, Pw = (Kh - n_bins % Kh) % Kh, (Kw - n_frames % Kw) % Kw
        padding_top = Ph // 2
        padding_bottom = Ph - padding_top
        padding_left = Pw // 2
        padding_right = Pw - padding_left

        input = F.pad(input, (padding_left, padding_right, padding_top, padding_bottom))
        
        x = self.dense_block(input)
        skip = x
        skip = F.pad(skip, (-padding_left, -padding_right, -padding_top, -padding_bottom))

        output = self.downsample2d(x)

        return output, skip

class UpSampleDenseBlock(nn.Module):
    """
    DenseBlock + up sample
    """
    def __init__(self, in_channels, skip_channels, growth_rate, kernel_size=(2,2), up_scale=(2,2), dilated=False, norm=True, nonlinear='relu', depth=None, eps=EPS):
        super().__init__()

        self.norm2d = choose_layer_norm('BN', in_channels, n_dims=2, eps=eps) # nn.BatchNorm2d
        self.upsample2d = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=up_scale, stride=up_scale)
        self.dense_block = DenseBlock(in_channels + skip_channels, growth_rate, kernel_size, dilated=dilated, norm=norm, nonlinear=nonlinear, depth=depth, eps=eps)
    
    def forward(self, input, skip):
        x = self.norm2d(input)
        x = self.upsample2d(x)

        _, _, H, W = x.size()
        _, _, H_skip, W_skip = skip.size()
        padding_height = H - H_skip
        padding_width = W - W_skip
        padding_top = padding_height // 2
        padding_bottom = padding_height - padding_top
        padding_left = padding_width // 2
        padding_right = padding_width - padding_left

        x = F.pad(x, (-padding_left, -padding_right, -padding_top, -padding_bottom))
        x = torch.cat([x, skip], dim=1)

        output = self.dense_block(x)

        return output

class DenseLSTMBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, kernel_size, depth=None, dilated=False, norm=True, nonlinear='relu', rnn_type='lstm', eps=EPS):
        """
        Args:
            in_channels <int>: # of input channels
            growth_rate <int> or <list<int>>: # of output channels
            kernel_size <int> or <tuple<int>>: Kernel size
            dilated <bool> or <list<bool>>: Applies dilated convolution.
            norm <bool> or <list<bool>>: Applies batch normalization.
            nonlinear <str> or <list<str>>: Applies nonlinear function.
            depth <int>: If `growth_rate` is given by list, len(growth_rate) must be equal to `depth`.
        """
        super().__init__()

        self.dense_block = DenseBlock(
            in_channels, growth_rate,
            kernel_size=kernel_size, depth=depth, dilated=dilated, norm=norm, nonlinear=nonlinear,
            eps=eps
        )
        self.rnn = choose_rnn(rnn_type)
    
    def forward(self, input):
        """
        Args:
            input: (batch_size, in_channels, H, W)
        Returns:
            output: (batch_size, out_channels, H, W), where `out_channels` is determined by ... 
        """

        x_dense = self.dense_block(input)
        x_rnn = self.rnn(input)
        output = x_dense + x_rnn

        return output

def _test_encoder():
    batch_size = 4
    n_bins, n_frames = 16, 64
    in_channels = 32

    growth_rate = [2, 3, 4]
    kernel_size = 3
    num_d2blocks = 2
    
    depth = [2, 2, 3]
    input = torch.randn(batch_size, in_channels, n_bins, n_frames)
    encoder = Encoder(in_channels, growth_rate, kernel_size, num_d2blocks=num_d2blocks, depth=depth)
    output, skip = encoder(input)

    print(encoder)
    print(input.size(), output.size())
    for _skip in skip:
        print(_skip.size())
    print()

    depth = 2
    encoder = Encoder(in_channels, growth_rate, kernel_size, num_d2blocks=num_d2blocks, depth=depth)
    output, skip = encoder(input)

    print(encoder)
    print(input.size(), output.size())

    for _skip in skip:
        print(_skip.size())

if __name__ == '__main__':
    torch.manual_seed(111)

    print('='*10, "DenseBlock", '='*10)
    _test_dense_block()
    print()

    print('='*10, "Encoder", '='*10)
    # _test_encoder()