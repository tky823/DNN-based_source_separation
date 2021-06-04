import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from models.transform import BandSplit
from models.gtu import GTU2d
from models.d2net import D2Block, CompressedD2Block

"""
Reference: D3Net: Densely connected multidilated DenseNet for music source separation
See https://arxiv.org/abs/2010.01733
"""

EPS=1e-12

class D3Net(nn.Module):
    def __init__(
        self, in_channels, num_features, growth_rate, bottleneck_channels, kernel_size, sections=[256,1344], scale=(2,2),
        num_d3blocks=5, num_d2blocks=3, depth=None, compressed_depth=None,
        growth_rate_d2block=None, kernel_size_d2block=None, depth_d2block=None,
        kernel_size_gated=None,
        norm=True, nonlinear='relu',
        eps=EPS, **kwargs
    ):
        super().__init__()

        self.in_channels, self.num_features = in_channels, num_features
        self.growth_rate = growth_rate
        self.bottleneck_channels = bottleneck_channels
        self.kernel_size = kernel_size
        self.sections = sections
        self.scale = scale
        self.num_d3blocks, self.num_d2blocks = num_d3blocks, num_d2blocks
        self.depth, self.compressed_depth = depth, compressed_depth

        self.growth_rate_d2block = growth_rate_d2block
        self.kernel_size_d2block = kernel_size_d2block
        self.depth_d2block = depth_d2block

        self.kernel_size_gated = _pair(kernel_size_gated)
        self.norm, self.nonlinear = norm, nonlinear,
        
        self.eps = eps

        self.band_split = BandSplit(sections=sections)

        net = {}
        self.bands = ['low', 'high', 'full']

        for key in self.bands:
            if compressed_depth is None:
                net[key] = D3NetBackbone(in_channels, num_features[key], growth_rate[key], bottleneck_channels, kernel_size[key], scale=scale[key], num_d3blocks=num_d3blocks[key], num_d2blocks=num_d2blocks[key], depth=depth[key], norm=norm, nonlinear=nonlinear, eps=eps, **kwargs)
            else:
                net[key] = D3NetBackbone(in_channels, num_features[key], growth_rate[key], bottleneck_channels, kernel_size[key], scale=scale[key], num_d3blocks=num_d3blocks[key], num_d2blocks=num_d2blocks[key], depth=depth[key], compressed_depth=compressed_depth[key], norm=norm, nonlinear=nonlinear, eps=eps, **kwargs)
        self.net = nn.ModuleDict(net)

        in_channels_d2block = len(self.bands[:-1]) * bottleneck_channels

        self.d2block = D2Block(in_channels_d2block, growth_rate_d2block, kernel_size_d2block, depth=depth_d2block, norm=norm, nonlinear=nonlinear, eps=eps)
        self.gated_conv2d = GTU2d(depth_d2block * growth_rate_d2block, in_channels, kernel_size=kernel_size_gated, stride=(1,1), padding=(1,1))

        self.num_parameters = self._get_num_parameters()

        warnings.warn("We will change the network architecture.", DeprecationWarning)
    
    def forward(self, input):
        stacked = []

        x = self.band_split(input)

        for idx, key in enumerate(self.bands[:-1]):
            _x = self.net[key](x[idx])
            stacked.append(_x)
        
        stacked = torch.cat(stacked, dim=2)
        
        key = self.bands[-1] # 'full'
        x = self.net[key](input)
        x = torch.cat([stacked, x], dim=1)
        x = self.d2block(x)
        
        output = self.gated_conv2d(x)

        return output
    
    def get_package(self):
        package = {
            'in_channels': self.in_channels,
            'num_features': self.num_features,
            'growth_rate': self.growth_rate,
            'bottleneck_channels': self.bottleneck_channels,
            'kernel_size': self.kernel_size,
            'sections': self.sections,
            'scale': self.scale,
            'num_d3blocks': self.num_d3blocks,
            'num_d2blocks': self.num_d2blocks,
            'depth': self.depth,
            'compressed_depth': self.compressed_depth,
            'growth_rate_d2block': self.growth_rate_d2block,
            'kernel_size_d2block': self.kernel_size_d2block,
            'depth_d2block': self.depth_d2block,
            'kernel_size_gated': self.kernel_size_gated,
            'norm': self.norm,
            'nonlinear': self.nonlinear,
            'eps': self.eps
        }
        
        return package
    
    @classmethod
    def build_model(cls, model_path):
        package = torch.load(model_path, map_location=lambda storage, loc: storage)
        
        in_channels, num_features = package['in_channels'], package['num_features']
        growth_rate = package['growth_rate']
        bottleneck_channels = package['bottleneck_channels']
        kernel_size = package['kernel_size']
        sections = package['sections']

        scale = package['scale']
        num_d3blocks, num_d2blocks = package['num_d3blocks'], package['num_d2blocks']
        depth, compressed_depth = package['depth'], package['compressed_depth']

        growth_rate_d2block = package['growth_rate_d2block']
        kernel_size_d2block = package['kernel_size_d2block']
        depth_d2block = package['depth_d2block']

        kernel_size_gated = package['kernel_size_gated']
        norm, nonlinear = package['norm'], package['nonlinear']
        eps = package['eps']

        model = cls(
            in_channels, num_features, growth_rate, bottleneck_channels, kernel_size,
            sections=sections, scale=scale,
            num_d3blocks=num_d3blocks, num_d2blocks=num_d2blocks, depth=depth, compressed_depth=compressed_depth,
            growth_rate_d2block=growth_rate_d2block, kernel_size_d2block=kernel_size_d2block, depth_d2block=depth_d2block,
            kernel_size_gated=kernel_size_gated,
            norm=norm, nonlinear=nonlinear,
            eps=eps
        )

        return model
    
    def _get_num_parameters(self):
        num_parameters = 0
        
        for p in self.parameters():
            if p.requires_grad:
                num_parameters += p.numel()
                
        return num_parameters

class D3NetBackbone(nn.Module):
    def __init__(self, in_channels, num_features, growth_rate, bottleneck_channels, kernel_size, scale=(2,2), num_d3blocks=5, num_d2blocks=3, depth=None, compressed_depth=None, norm=True, nonlinear='relu', eps=EPS):
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
        self.bottleneck_conv2d = nn.Conv2d(num_features, bottleneck_channels, kernel_size=(1,1), stride=(1,1))
        
    def forward(self, input):
        """
        Returns:
            output (batch_size, bottleneck_channels, H, W)
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

        output = self.bottleneck_conv2d(x)

        return output

class DownD3Block(nn.Module):
    """
    D3Block + down sample
    """
    def __init__(self, in_channels, growth_rate, kernel_size, down_scale=(2,2), num_blocks=3, depth=None, compressed_depth=None, norm=True, nonlinear='relu', eps=EPS):
        super().__init__()

        self.down_scale = _pair(down_scale)

        self.downsample2d = nn.AvgPool2d(kernel_size=self.down_scale, stride=self.down_scale)
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
        _, _, n_bins, n_frames = input.size()

        Kh, Kw = self.down_scale
        Ph, Pw = (Kh - n_bins % Kh) % Kh, (Kw - n_frames % Kw) % Kw
        padding_up = Ph // 2
        padding_bottom = Ph - padding_up
        padding_left = Pw // 2
        padding_right = Pw - padding_left

        input = F.pad(input, (padding_left, padding_right, padding_up, padding_bottom))

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
            input (batch_size, in_channels, H, W)
            output 
                (batch_size, num_blocks * sum(growth_rate), H*Sh, W*Sw) if type(growth_rate) is list<int>
                or (batch_size, num_blocks * depth * growth_rate, H*Sh, W*Sw) if type(growth_rate) is int
                where Sh and Sw are scale factors
        """
        x = self.upsample2d(input)
        if skip is not None:
            _, _, H, W = x.size()
            _, _, H_skip, W_skip = skip.size()
            padding_height, padding_width = H - H_skip, W - W_skip
            padding_up = padding_height // 2
            padding_bottom = padding_height - padding_up
            padding_left = padding_width // 2
            padding_right = padding_width - padding_left
            x = F.pad(x, (-padding_left, -padding_right, -padding_up, -padding_bottom))
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
    in_channels, num_features, growth_rate, bottleneck_channels = 3, 32, [2, 3, 4, 5, 4, 3, 2], 32
    depth = [4, 4, 4, 4, 3, 3, 2]
    num_d3blocks, num_d2blocks = 7, [2, 2, 2, 2, 2, 1, 3]

    input = torch.randn(batch_size, in_channels, H, W)

    model = D3NetBackbone(in_channels, num_features, growth_rate, bottleneck_channels, kernel_size=(3,3), scale=(2,2), num_d3blocks=num_d3blocks, num_d2blocks=num_d2blocks, depth=depth)
    print(model)
    output = model(input)
    print(input.size(), output.size())

def _test_d3net():
    torch.manual_seed(111)
    
    batch_size = 4
    sections = [64, 128]
    H, W = sum(sections), 128
    
    in_channels, num_features, growth_rate, bottleneck_channels = 2, {'low': 32, 'high': 8, 'full': 8}, {'low': [3, 4, 5, 4, 3], 'high': [3, 4, 3], 'full': [3, 4, 3]}, 8
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
        in_channels, num_features, growth_rate, bottleneck_channels, kernel_size=kernel_size, sections=sections, scale=scale,
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
        in_channels, num_features, growth_rate, bottleneck_channels, kernel_size=kernel_size, sections=sections, scale=scale,
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
    sections = [256, 1344]
    H, W = sum(sections), 256
    in_channels, num_features, growth_rate, bottleneck_channels = 2, {'low': 32, 'high': 8, 'full': 32}, {'low': [16, 18, 20, 22, 20, 18, 16], 'high': [2, 2, 2, 2, 2, 2, 2], 'full': [13, 14, 15, 16, 17, 16, 14, 12, 11]}, 8
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
        in_channels, num_features, growth_rate, bottleneck_channels, kernel_size=kernel_size, sections=sections, scale=scale,
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