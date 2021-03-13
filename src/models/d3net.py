from itertools import compress
import torch
import torch.nn as nn

from models.transform import BandSplit
from models.d2net import D2Block, CompressedD2Block

"""
Reference: D3Net: Densely connected multidilated DenseNet for music source separation
See https://arxiv.org/abs/2010.01733
"""

EPS=1e-12

class D3Net(nn.Module):
    def __init__(
        self, in_channels, num_features, growth_rate, kernel_size, sections=[256,1344], scale=(2,2),
        num_d3blocks=5, num_d2blocks=3, depth=None, compressed_depth=None,
        growth_rate_d2block=None, kernel_size_d2block=None, depth_d2block=None,
        kernel_size_gated=None,
        norm=True, nonlinear='relu',
        eps=EPS, **kwargs
    ):
        super().__init__()

        self.band_split = BandSplit(sections=sections)

        in_channels_d2block = 0
        net = {}
        self.bands = ['low', 'high', 'full']

        for key in self.bands:
            if compressed_depth is None:
                net[key] = D3NetBackbone(in_channels, num_features[key], growth_rate[key], kernel_size[key], scale=scale[key], num_d3blocks=num_d3blocks[key], num_d2blocks=num_d2blocks[key], depth=depth[key], norm=norm, nonlinear=nonlinear, eps=eps, **kwargs)
            else:
                net[key] = D3NetBackbone(in_channels, num_features[key], growth_rate[key], kernel_size[key], scale=scale[key], num_d3blocks=num_d3blocks[key], num_d2blocks=num_d2blocks[key], depth=depth[key], compressed_depth=compressed_depth[key], norm=norm, nonlinear=nonlinear, eps=eps, **kwargs)

        if compressed_depth is None or compressed_depth[key] is None:
            in_channels_d2block = 2 * num_d2blocks[key][-1] * depth[key][-1] * growth_rate[key][-1]
        else:
            in_channels_d2block = 2 * num_d2blocks[key][-1] * compressed_depth[key][-1] * growth_rate[key][-1]

        self.net = nn.ModuleDict(net)

        self.d2block = D2Block(in_channels_d2block, growth_rate_d2block, kernel_size_d2block, depth=depth_d2block, norm=norm, nonlinear=nonlinear, eps=eps)
        self.gated_conv2d = nn.Conv2d(depth_d2block * growth_rate_d2block, in_channels, kernel_size=kernel_size_gated, stride=(1,1), padding=(1,1))

        self.eps = eps

    
    def forward(self, input):
        stacked = []

        x = self.band_split(input)

        for idx, key in enumerate(self.bands[:-1]):
            _x = self.net[key](x[idx])
            stacked.append(_x)
        
        stacked = torch.cat(stacked, dim=2)
        
        key = self.bands[-1] # 'full'
        x = self.net[key](input)
        print(stacked.size(), x.size())
        x = torch.cat([stacked, x], dim=1)
        x = self.d2block(x)
        output = self.gated_conv2d(x)

        return output

class D3NetBackbone(nn.Module):
    def __init__(self, in_channels, num_features, growth_rate, kernel_size, scale=(2,2), num_d3blocks=5, num_d2blocks=3, depth=None, compressed_depth=None, norm=True, nonlinear='relu', eps=EPS):
        super().__init__()

        assert num_d3blocks % 2 == 1, "`num_d3blocks` must be odd number"
        self.num_stacks = num_d3blocks // 2 + 1

        encoder = []
        decoder = []

        self.conv2d = nn.Conv2d(in_channels, num_features, kernel_size=(3,3), stride=(1,1), padding=(1,1))

        if compressed_depth is None:
            encoder.append(D3Block(num_features, growth_rate[0], kernel_size, num_blocks=num_d2blocks[0], depth=depth[0], norm=norm, nonlinear=nonlinear, eps=eps))
            num_features = num_d2blocks[0] * depth[0] * growth_rate[0]
        else:
            # TODO
            encoder.append(D3Block(num_features, growth_rate[0], kernel_size, num_blocks=num_d2blocks[0], depth=depth[0], compressed_depth=compressed_depth[0], norm=norm, nonlinear=nonlinear, eps=eps))
            num_features = num_d2blocks[0] * compressed_depth[0] * growth_rate[0]
        

        for idx in range(1, self.num_stacks):
            if compressed_depth is None:
                encoder.append(DownD3Block(num_features, growth_rate[idx], kernel_size, down_scale=scale, num_blocks=num_d2blocks[idx], depth=depth[idx], norm=norm, nonlinear=nonlinear, eps=eps))
                num_features = num_d2blocks[idx] * depth[idx] * growth_rate[idx]
            else:
                encoder.append(DownD3Block(num_features, growth_rate[idx], kernel_size, down_scale=scale, num_blocks=num_d2blocks[idx], depth=depth[idx], compressed_depth=compressed_depth[idx], norm=norm, nonlinear=nonlinear, eps=eps))
                num_features = num_d2blocks[idx] * compressed_depth[idx] * growth_rate[idx]
        
        for idx in range(self.num_stacks, num_d3blocks):
            skip_idx = num_d3blocks - idx - 1

            if compressed_depth is None:
                skip_channels = num_d2blocks[skip_idx] * depth[skip_idx] * growth_rate[skip_idx]
                decoder.append(UpD3Block(num_features, growth_rate[idx], kernel_size, up_scale=scale, skip_channels=skip_channels, num_blocks=num_d2blocks[idx], depth=depth[idx], norm=norm, nonlinear=nonlinear, eps=eps))    
                num_features = num_d2blocks[idx] * depth[idx] * growth_rate[idx]
            else:
                skip_channels = num_d2blocks[skip_idx] * compressed_depth[skip_idx] * growth_rate[skip_idx]
                decoder.append(UpD3Block(num_features, growth_rate[idx], kernel_size, up_scale=scale, skip_channels=skip_channels, num_blocks=num_d2blocks[idx], depth=depth[idx], compressed_depth=compressed_depth[idx], norm=norm, nonlinear=nonlinear, eps=eps))    
                num_features = num_d2blocks[idx] * compressed_depth[idx] * growth_rate[idx]
        
        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Sequential(*decoder)
        
    def forward(self, input):
        """
        Returns:
            ?? output (batch_size, num_d2blocks[num_d3blocks - 2] * depth[num_d3blocks - 2] * growth_rate[num_d3blocks - 2], H, W)
        """
        x = self.conv2d(input)

        skips = []
        skips.append(x)

        for idx in range(self.num_stacks):
            x = self.encoder[idx](x)
            skips.append(x)

        for idx in range(self.num_stacks - 1):
            skip_idx = self.num_stacks - idx - 1
            skip = skips[skip_idx]
            x = self.decoder[idx](x, skip=skip)

        output = x

        return output

class DownD3Block(nn.Module):
    """
    D3Block + down sample
    """
    def __init__(self, in_channels, growth_rate, kernel_size, down_scale=(2,2), num_blocks=3, depth=None, compressed_depth=None, norm=True, nonlinear='relu', eps=EPS):
        super().__init__()

        self.downsample2d = nn.AvgPool2d(kernel_size=down_scale, stride=down_scale)
        self.d3block = D3Block(in_channels, growth_rate, kernel_size, num_blocks=num_blocks, depth=depth, compressed_depth=compressed_depth, norm=norm, nonlinear=nonlinear, eps=eps)
    
    def forward(self, input):
        """
        Args:
            input (batch_size, in_channels, H, W)
            output:
                (batch_size, num_blocks * sum(growth_rate), H_down, W_down) if type(growth_rate) is list<int>
                or (batch_size, num_blocks * depth * growth_rate, H_down, W_down) if type(growth_rate) is int
                where H_down = H // down_scale[0] and W_down = W // down_scale[1]
        """
        x = self.downsample2d(input)
        output = self.d3block(x)

        return output

class UpD3Block(nn.Module):
    """
    D3Block + up sample
    """
    def __init__(self, in_channels, growth_rate, kernel_size, up_scale=(2,2), skip_channels=None, num_blocks=3, depth=None, compressed_depth=None, norm=True, nonlinear='relu', eps=EPS):
        super().__init__()

        self.skip_channels = skip_channels

        self.upsample2d = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=up_scale, stride=up_scale, groups=in_channels, bias=False)
        if skip_channels is not None:
            self.d3block = D3Block(in_channels + skip_channels, growth_rate, kernel_size, num_blocks=num_blocks, depth=depth, compressed_depth=compressed_depth, norm=norm, nonlinear=nonlinear, eps=eps)
        else:
            self.d3block = D3Block(in_channels, growth_rate, kernel_size, num_blocks=num_blocks, depth=depth, compressed_depth=compressed_depth, norm=norm, nonlinear=nonlinear, eps=eps)
    
    def forward(self, input, skip=None):
        """
        Args:
            input (batch_size, in_channels, n_bins, n_frames)
            output 
                (batch_size, num_blocks * sum(growth_rate), n_bins, n_frames) if type(growth_rate) is list<int>
                or (batch_size, num_blocks * depth * growth_rate, n_bins, n_frames) if type(growth_rate) is int
        """
        x = self.upsample2d(input)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        output = self.d3block(x)

        return output

class D3Block(nn.Module):
    def __init__(self, in_channels, growth_rate, kernel_size, num_blocks=3, depth=None, compressed_depth=None, norm=True, nonlinear='relu', eps=EPS):
        super().__init__()

        self.num_blocks = num_blocks

        if type(growth_rate) is int:
            growth_rate = [
                growth_rate for _ in range(num_blocks)
            ]
        elif type(growth_rate) is list:
            pass
        else:
            raise ValueError("Not support `growth_rate`={}".format(growth_rate))
            
        if depth is None:
            depth = [
                None for _ in range(num_blocks)
            ]
        elif type(depth) is int:
            depth = [
                depth for _ in range(num_blocks)
            ]
        
        if compressed_depth is not None:
            if type(compressed_depth) is int:
                compressed_depth = [
                    compressed_depth for _ in range(num_blocks)
                ]
            elif type(compressed_depth) is list:
                pass
            else:
                raise ValueError("Not support `compressed_depth`={}".format(compressed_depth))

        net = []

        for idx in range(num_blocks):
            if compressed_depth is None:
                net.append(D2Block(in_channels, growth_rate[idx], kernel_size, depth=depth[idx], norm=norm, nonlinear=nonlinear, eps=eps))
                in_channels += growth_rate[idx] * depth[idx]
            else:
                net.append(CompressedD2Block(in_channels, growth_rate[idx], kernel_size, depth=depth[idx], compressed_depth=compressed_depth[idx], norm=norm, nonlinear=nonlinear, eps=eps))
                in_channels += growth_rate[idx] * compressed_depth[idx]
        
        self.net = nn.Sequential(*net)

        self.eps = eps
    
    def forward(self, input):
        """
        Args:
            input (batch_size, in_channels, n_bins, n_frames)
            output
                (batch_size, sum(growth_rate), n_bins, n_frames)
                or (batch_size, num_blocks*growth_rate, n_bins, n_frames)
        """
        x = input
        stacked = []

        stacked.append(input)

        for idx in range(self.num_blocks):
            if idx != 0:
                x = torch.cat(stacked, dim=1)
            x = self.net[idx](x)
            stacked.append(x)
        
        output = torch.cat(stacked[1:], dim=1)

        return output

def _test_d3block():
    torch.manual_seed(111)
    
    batch_size = 4
    H, W = 16, 32
    in_channels, growth_rate = 3, 2
    depth = 4
    scale = (2, 2)
    num_blocks = 3

    input = torch.randn(batch_size, in_channels, H, W)

    model = D3Block(in_channels, growth_rate, kernel_size=(3,3), num_blocks=num_blocks, depth=depth)
    print(model)
    output = model(input)
    print(input.size(), output.size())
    print()

    model = DownD3Block(in_channels, growth_rate, kernel_size=(3,3), down_scale=scale, num_blocks=num_blocks, depth=depth)
    print(model)
    output = model(input)
    print(input.size(), output.size())
    print()

    model = UpD3Block(in_channels, growth_rate, kernel_size=(3,3), up_scale=scale, num_blocks=num_blocks, depth=depth)
    print(model)
    output = model(input)
    print(input.size(), output.size())

def _test_backbone():
    torch.manual_seed(111)
    
    batch_size = 4
    H, W = 64, 128
    in_channels, num_features, growth_rate = 3, 32, [2, 3, 4, 5, 4, 3, 2]
    depth = [4, 4, 4, 4, 3, 3, 2]
    num_d3blocks, num_d2blocks = 7, [2, 2, 2, 2, 2, 1, 3]

    input = torch.randn(batch_size, in_channels, H, W)

    model = D3NetBackbone(in_channels, num_features, growth_rate, kernel_size=(3,3), scale=(2,2), num_d3blocks=num_d3blocks, num_d2blocks=num_d2blocks, depth=depth)
    print(model)
    output = model(input)
    print(input.size(), output.size())

def _test_d3net():
    torch.manual_seed(111)
    
    batch_size = 4
    sections = [64, 128]
    H, W = sum(sections), 128
    
    in_channels, num_features, growth_rate = 2, {'low': 32, 'high': 8, 'full': 8}, {'low': [3, 4, 5, 4, 3], 'high': [3, 4, 3], 'full': [3, 4, 3]}
    kernel_size = {'low': (3, 3), 'high': (3, 3), 'full': (3, 3)}
    scale = {'low': (2,2), 'high': (2,2), 'full': (2,2)}
    depth, compressed_depth = {'low': [4, 4, 4, 3, 3], 'high': [4, 4, 3], 'full': [4, 4, 3]}, {'low': [2, 2, 2, 1, 1], 'high': [2, 2, 1], 'full': [2, 2, 1]}
    num_d3blocks, num_d2blocks = {'low': 5, 'high': 3, 'full': 3}, {'low': [2, 2, 2, 2, 2], 'high': [2, 2, 2], 'full': [2, 2, 2]}

    kernel_size_d2block = (3, 3)
    growth_rate_d2block = 1
    depth_d2block = 2

    kernel_size_gated = (3, 3)

    input = torch.randn(batch_size, in_channels, H, W)

    print("-"*10, "D3Net w/o compression", "-"*10)
    model = D3Net(
        in_channels, num_features, growth_rate, kernel_size=kernel_size, sections=sections, scale=scale,
        num_d3blocks=num_d3blocks, num_d2blocks=num_d2blocks, depth=depth,
        growth_rate_d2block=growth_rate_d2block, kernel_size_d2block=kernel_size_d2block, depth_d2block=depth_d2block,
        kernel_size_gated=kernel_size_gated
    )
    print(model)
    output = model(input)
    print(input.size(), output.size())
    print()

    print("-"*10, "D3Net w/ compression", "-"*10)
    model = D3Net(
        in_channels, num_features, growth_rate, kernel_size=kernel_size, sections=sections, scale=scale,
        num_d3blocks=num_d3blocks, num_d2blocks=num_d2blocks, depth=depth, compressed_depth=compressed_depth,
        growth_rate_d2block=growth_rate_d2block, kernel_size_d2block=kernel_size_d2block, depth_d2block=depth_d2block,
        kernel_size_gated=kernel_size_gated
    )
    print(model)
    output = model(input)
    print(input.size(), output.size())

def _test_d3net_paper():
    torch.manual_seed(111)
    
    batch_size = 4
    sections = [256, 1344, 449]
    H, W = sum(sections), 256
    in_channels, num_features, growth_rate = 2, {'low': 32, 'high': 8, 'full': 32}, {'low': [16, 18, 20, 22, 20, 18, 16], 'high': [2, 2, 2, 2, 2, 2, 2], 'full': [13, 14, 15, 16, 17, 16, 14, 12, 11]}
    kernel_size = {'low': (3, 3), 'high': (3, 3), 'full': (3, 3)}
    scale = {'low': (2,2), 'high': (2,2), 'full': (2,2)}
    depth, compressed_depth = {'low': [5, 5, 5, 5, 4, 4, 4], 'high': [1, 1, 1, 1, 1, 1, 1], 'full': [4, 5, 6, 7, 8, 6, 5, 4, 4]}, {'low': [2, 2, 2, 2, 2, 2, 2], 'high': [1, 1, 1, 1, 1, 1, 1], 'full': [2, 2, 2, 2, 2, 2, 2, 2, 2]}
    num_d3blocks, num_d2blocks = {'low': 7, 'high': 7, 'full': 9}, {'low': [2, 2, 2, 2, 2, 2, 2], 'high': [1, 1, 1, 1, 1, 1, 1], 'full': [2, 2, 2, 2, 2, 2, 2, 2, 2]}

    kernel_size_d2block = (3, 3)
    growth_rate_d2block = 12
    depth_d2block = 3

    kernel_size_gated = (3, 3)

    input = torch.randn(batch_size, in_channels, H, W)

    model = D3Net(
        in_channels, num_features, growth_rate, kernel_size=kernel_size, sections=sections, scale=scale,
        num_d3blocks=num_d3blocks, num_d2blocks=num_d2blocks, depth=depth, compressed_depth=compressed_depth,
        growth_rate_d2block=growth_rate_d2block, kernel_size_d2block=kernel_size_d2block, depth_d2block=depth_d2block,
        kernel_size_gated=kernel_size_gated
    )
    print(model)
    output = model(input)
    print(input.size(), output.size())

if __name__ == '__main__':
    _test_d3block()
    print()

    _test_backbone()
    print()

    _test_d3net()
    print()

    _test_d3net_paper()