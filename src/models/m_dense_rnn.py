import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from utils.m_densenet import choose_layer_norm
from utils.dense_rnn import choose_dense_rnn_block
from models.m_densenet import DownSampleDenseBlock, UpSampleDenseBlock, DenseBlock
from models.dense_rnn import RNNBlock

"""
Reference: MMDenseLSTM: An efficient combination of convolutional and recurrent neural networks for audio source separation
See https://ieeexplore.ieee.org/document/8521383
"""

FULL = 'full'
EPS = 1e-12

class MDenseRNNBackbone(nn.Module):
    def __init__(self, in_channels, num_features, growth_rate, hidden_channels, kernel_size, n_bins=None, scale=(2,2), dilated=False, norm=True, nonlinear='relu', causal=False, depth=None, rnn_type='rnn', rnn_position='parallel', out_channels=None, eps=EPS):
        """
        Args:
            in_channels <int>
            num_features <int>
            growth_rate <list<int>>: `len(growth_rate)` must be an odd number.
            hidden_channels <list<int>>: `len(hidden_channels) = len(growth_rate)`.
            kernel_size <int> or <tuple<int>>
            scale <int> or <list<int>>: Upsampling and Downsampling scale
            dilated <list<bool>>
            norm <list<bool>>
            nonlinear <list<str>>
        """
        super().__init__()

        assert len(growth_rate) % 2 == 1, "`len(growth_rate)` must be an odd number."

        kernel_size = _pair(kernel_size)
        scale = _pair(scale)
        num_encoder_blocks = len(growth_rate) // 2

        # Network
        self.conv2d = nn.Conv2d(in_channels, num_features, kernel_size, stride=(1,1))

        encoder, decoder = [], []
        encoder = Encoder(
            num_features, growth_rate[:num_encoder_blocks], hidden_channels=hidden_channels[:num_encoder_blocks],
            kernel_size=kernel_size, down_scale=scale,
            dilated=dilated[:num_encoder_blocks], norm=norm[:num_encoder_blocks], nonlinear=nonlinear[:num_encoder_blocks],
            causal=causal,
            depth=depth[:num_encoder_blocks],
            eps=eps
        )

        _in_channels, _growth_rate = growth_rate[num_encoder_blocks - 1], growth_rate[num_encoder_blocks]
        _n_bins = n_bins
        n_bins_detail = [n_bins]

        for _ in range(num_encoder_blocks):
            remain = (scale[0] - (_n_bins % scale[0])) % scale[0]
            if remain > 0:
                _n_bins //= scale[0]
                _n_bins += 1
            else:
                _n_bins //= scale[0]
            n_bins_detail.append(_n_bins)

        if hidden_channels[num_encoder_blocks] <= 0:
            bottleneck_dense_block = DenseBlock(
                _in_channels, _growth_rate,
                kernel_size=kernel_size,
                dilated=dilated[num_encoder_blocks], norm=norm[num_encoder_blocks], nonlinear=nonlinear[num_encoder_blocks],
                depth=depth[num_encoder_blocks],
                eps=eps
            )
        elif depth[num_encoder_blocks] == 0:
            bottleneck_dense_block = RNNBlock(
                _in_channels, hidden_channels[num_encoder_blocks],
                n_bins=n_bins_detail[-1],
                causal=causal, rnn_type=rnn_type
            )
        elif depth[num_encoder_blocks] < 0:
            raise NotImplementedError("Invalid depth is specified.")
        else:
            bottleneck_dense_block = choose_dense_rnn_block(
                rnn_type, rnn_position,
                _in_channels, _growth_rate, hidden_channels[num_encoder_blocks],
                kernel_size=kernel_size,
                n_bins=n_bins_detail[-1],
                dilated=dilated[num_encoder_blocks], norm=norm[num_encoder_blocks], nonlinear=nonlinear[num_encoder_blocks],
                causal=causal,
                depth=depth[num_encoder_blocks],
                eps=eps
            )

        _in_channels = bottleneck_dense_block.out_channels
        skip_channels = encoder.skip_channels
        n_bins_detail = n_bins_detail[num_encoder_blocks - 1::-1]

        decoder = Decoder(
            _in_channels, skip_channels[::-1], growth_rate[num_encoder_blocks + 1:], hidden_channels=hidden_channels[num_encoder_blocks + 1:],
            kernel_size=kernel_size, n_bins=n_bins_detail, up_scale=scale,
            dilated=dilated[num_encoder_blocks + 1:], depth=depth[num_encoder_blocks + 1:], norm=norm[num_encoder_blocks + 1:], nonlinear=nonlinear[num_encoder_blocks + 1:],
            causal=causal,
            rnn_type=rnn_type, rnn_position=rnn_position,
            eps=eps
        )
        
        self.encoder = encoder
        self.bottleneck_conv2d = bottleneck_dense_block
        self.decoder = decoder

        if out_channels is not None:
            _in_channels = decoder.out_channels

            net = []
            norm2d = choose_layer_norm('BN', _in_channels, n_dims=2, eps=eps) # nn.BatchNorm2d
            net.append(norm2d)
            net.append(nn.Conv2d(_in_channels, out_channels, kernel_size=(1,1), stride=(1,1)))

            self.pointwise_conv2d = nn.Sequential(*net)
        else:
            self.pointwise_conv2d = None

        self.kernel_size = kernel_size
        self.out_channels = out_channels
    
    def forward(self, input):
        Kh, Kw = self.kernel_size
        Ph, Pw = Kh - 1, Kw - 1
        padding_top = Ph // 2
        padding_bottom = Ph - padding_top
        padding_left = Pw // 2
        padding_right = Pw - padding_left

        input = F.pad(input, (padding_left, padding_right, padding_top, padding_bottom))

        x = self.conv2d(input)
        x, skip = self.encoder(x)
        x = self.bottleneck_conv2d(x)
        x = self.decoder(x, skip[::-1])

        if self.pointwise_conv2d:
            output = self.pointwise_conv2d(x)
        else:
            output = x
        
        return output

class Encoder(nn.Module):
    def __init__(self, in_channels, growth_rate, hidden_channels, kernel_size, down_scale=(2,2), dilated=False, norm=True, nonlinear='relu', causal=False, depth=None, eps=EPS):
        """
        Args:
            in_channels <int>: 
            growth_rate <list<int>>:
            kernel_size <tuple<int>> or <int>:
            hidden_channels <list<int>>:
            dilated <list<bool>> or <bool>:
            norm <list<bool>> or <bool>:
            nonlinear <list<str>> or <str>:
            depth <list<int>> or <int>:
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
        skip_channels = []
        net = []

        _in_channels = in_channels

        for idx in range(num_dense_blocks):
            if hidden_channels[idx] <= 0:
                downsample_block = DownSampleDenseBlock(
                    _in_channels, growth_rate[idx],
                    kernel_size=kernel_size, down_scale=down_scale,
                    dilated=dilated[idx], norm=norm[idx], nonlinear=nonlinear[idx],
                    depth=depth[idx],
                    eps=eps
                )
                skip_channels.append(downsample_block.out_channels)
            else:
                raise NotImplementedError("Not support DownSampleDenseRNNBlock now.")

            net.append(downsample_block)
            _in_channels = skip_channels[-1]

        self.net = nn.Sequential(*net)

        self.skip_channels = skip_channels
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

class Decoder(nn.Module):
    def __init__(self, in_channels, skip_channels, growth_rate, hidden_channels, kernel_size, n_bins=None, up_scale=(2,2), dilated=False, norm=True, nonlinear='relu', causal=False, depth=None, rnn_type='rnn', rnn_position='parallel', eps=EPS):
        """
        Args:
            in_channels <int>: 
            skip_channels <list<int>>:
            growth_rate <list<int>>:
            hidden_channels <list<int>>:
            kernel_size <tuple<int>> or <int>:
            n_bins <int> or <list<int>>:
            up_scale <tuple<int>>
            dilated <list<bool>> or <bool>:
            norm <list<bool>> or <bool>:
            nonlinear <list<str>> or <str>:
            depth <list<int>> or <int>:
        """
        super().__init__()

        if type(growth_rate) is list:
            num_dense_blocks = len(growth_rate)
        else:
            # TODO: implement
            raise ValueError("`growth_rate` must be list.")

        if type(hidden_channels) is int:
            hidden_channels = [hidden_channels] * num_dense_blocks
        elif type(hidden_channels) is list:
            assert num_dense_blocks == len(hidden_channels), "Invalid length of `hidden_channels`"
        else:
            raise ValueError("`hidden_channels` must be list.")

        if type(n_bins) is int:
            _n_bins = n_bins
            n_bins = []
            for _ in range(num_dense_blocks):
                _n_bins *= up_scale[0]
                n_bins.append(_n_bins)
        elif type(n_bins) is list:
            assert num_dense_blocks == len(n_bins), "Invalid length of `n_bins`"
        else:
            raise ValueError("`hidden_channels` must be list.")

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
            if hidden_channels[idx] <= 0:
                upsample_block = UpSampleDenseBlock(
                    _in_channels, skip_channels[idx], growth_rate[idx],
                    kernel_size=kernel_size, up_scale=up_scale,
                    dilated=dilated[idx], norm=norm[idx], nonlinear=nonlinear[idx],
                    depth=depth[idx],
                    eps=eps
                )
            else:
                upsample_block = UpSampleDenseRNNBlock(
                    _in_channels, skip_channels[idx], growth_rate[idx], hidden_channels=hidden_channels[idx],
                    kernel_size=kernel_size, n_bins=n_bins[idx], up_scale=up_scale,
                    dilated=dilated[idx], norm=norm[idx], nonlinear=nonlinear[idx],
                    causal=causal,
                    depth=depth[idx],
                    rnn_type=rnn_type, rnn_position=rnn_position,
                    eps=eps
                )

            net.append(upsample_block)
            _in_channels = upsample_block.out_channels

        self.net = nn.Sequential(*net)

        self.out_channels = upsample_block.out_channels
        self.num_dense_blocks = num_dense_blocks

    def forward(self, input, skip):
        num_dense_blocks = self.num_dense_blocks

        x = input

        for idx in range(num_dense_blocks):
            x_skip = skip[idx]
            x = self.net[idx](x, x_skip)

        output = x

        return output

class UpSampleDenseRNNBlock(nn.Module):
    """
    DenseRNNBlock + up sample
    """
    def __init__(self, in_channels, skip_channels, growth_rate, hidden_channels, kernel_size=(2,2), n_bins=None, up_scale=(2,2), dilated=False, norm=True, nonlinear='relu', causal=False, depth=None, rnn_type='rnn', rnn_position='parallel', eps=EPS):
        super().__init__()

        self.norm2d = choose_layer_norm('BN', in_channels, n_dims=2, eps=eps) # nn.BatchNorm2d
        self.upsample2d = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=up_scale, stride=up_scale)
        self.dense_rnn_block = choose_dense_rnn_block(rnn_type, rnn_position, in_channels + skip_channels, growth_rate, hidden_channels, kernel_size, n_bins=n_bins, dilated=dilated, norm=norm, nonlinear=nonlinear, causal=causal, depth=depth, eps=eps)

        self.out_channels = self.dense_rnn_block.out_channels

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

        output = self.dense_rnn_block(x)

        return output

def _test_m_dense_rnn_backbone():
    batch_size = 4
    n_bins, n_frames = 16, 64
    in_channels, num_features = 2, 32

    growth_rate = [2, 3, 4, 4, 2]
    hidden_channels = [0, 0, 1, 3, 2] # len(growth_rate)
    kernel_size = 3

    dilated = [True, True, True, True, True]
    norm = [True, True, True, True, True]
    nonlinear = ['relu', 'relu', 'relu', 'relu', 'relu']
    depth = [3, 3, 4, 2, 2]
    input = torch.randn(batch_size, in_channels, n_bins, n_frames)

    model = MDenseRNNBackbone(
        in_channels, num_features, growth_rate,
        hidden_channels,
        kernel_size,
        n_bins=n_bins,
        dilated=dilated, norm=norm, nonlinear=nonlinear,
        depth=depth
    )

    print(model)

    output = model(input)

    print(input.size(), output.size())

if __name__ == '__main__':
    torch.manual_seed(111)

    print('='*10, "MMDenseRNN backbone", '='*10)
    _test_m_dense_rnn_backbone()