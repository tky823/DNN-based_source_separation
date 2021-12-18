import torch
import torch.nn as nn
import torch.nn.quantized as nnq

from models.m_densenet import ConvBlock2d, QuantizableConvBlock2d

EPS = 1e-12

class D2BlockFixedDilation(nn.Module):
    def __init__(self, in_channels, growth_rate, kernel_size, dilation=1, norm=True, nonlinear='relu', depth=None, eps=EPS):
        """
        Args:
            in_channels <int>: # of input channels
            growth_rate <int> or <list<int>>: # of output channels
            kernel_size <int> or <tuple<int>>: Kernel size
            dilation <int>: Dilataion of dilated convolution.
            norm <bool> or <list<bool>>: Applies batch normalization.
            nonlinear <str> or <list<str>>: Applies nonlinear function.
            depth <int>: If `growth_rate` is given by list, len(growth_rate) must be equal to `depth`.
        """
        super().__init__()

        if type(growth_rate) is int:
            assert depth is not None, "Specify `depth`"
            growth_rate = [growth_rate] * depth
        elif type(growth_rate) is list:
            if depth is not None:
                assert depth == len(growth_rate), "`depth` is different from `len(growth_rate)`"
            depth = len(growth_rate)
        else:
            raise ValueError("Not support growth_rate={}".format(growth_rate))

        if not type(dilation) is int:
            raise ValueError("Not support dilation={}".format(dilation))

        if type(norm) is bool:
            assert depth is not None, "Specify `depth`"
            norm = [norm] * depth
        elif type(norm) is list:
            if depth is not None:
                assert depth == len(norm), "`depth` is different from `len(norm)`"
            depth = len(norm)
        else:
            raise ValueError("Not support norm={}".format(norm))

        if type(nonlinear) is bool or type(nonlinear) is str:
            assert depth is not None, "Specify `depth`"
            nonlinear = [nonlinear] * depth
        elif type(nonlinear) is list:
            if depth is not None:
                assert depth == len(nonlinear), "`depth` is different from `len(nonlinear)`"
            depth = len(nonlinear)
        else:
            raise ValueError("Not support nonlinear={}".format(nonlinear))

        self.growth_rate = growth_rate
        self.depth = depth

        net = []
        _in_channels = in_channels - sum(growth_rate)

        for idx in range(depth):
            if idx == 0:
                _in_channels = in_channels
            else:
                _in_channels = growth_rate[idx - 1]
            _out_channels = sum(growth_rate[idx:])

            conv_block = ConvBlock2d(_in_channels, _out_channels, kernel_size=kernel_size, stride=1, dilation=dilation, norm=norm[idx], nonlinear=nonlinear[idx], eps=eps)
            net.append(conv_block)

        self.net = nn.Sequential(*net)

    def forward(self, input):
        """
        Args:
            input: (batch_size, in_channels, H, W)
        Returns:
            output: (batch_size, out_channels, H, W), where out_channels = growth_rate[-1].
        """
        growth_rate, depth = self.growth_rate, self.depth

        x_residual = 0

        for idx in range(depth):
            if idx == 0:
                x = input
            else:
                _in_channels = growth_rate[idx - 1]
                sections = [_in_channels, sum(growth_rate[idx:])]
                x, x_residual = torch.split(x_residual, sections, dim=1)

            x = self.net[idx](x)
            x_residual = x_residual + x

        output = x_residual

        return output

class D2Block(nn.Module):
    def __init__(self, in_channels, growth_rate, kernel_size, dilated=True, norm=True, nonlinear='relu', depth=None, eps=EPS):
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

        if type(growth_rate) is int:
            assert depth is not None, "Specify `depth`"
            growth_rate = [growth_rate] * depth
        elif type(growth_rate) is list:
            if depth is not None:
                assert depth == len(growth_rate), "`depth` is different from `len(growth_rate)`"
            depth = len(growth_rate)
        else:
            raise ValueError("Not support growth_rate={}".format(growth_rate))

        if type(dilated) is bool:
            assert depth is not None, "Specify `depth`"
            dilated = [dilated] * depth
        elif type(dilated) is list:
            if depth is not None:
                assert depth == len(dilated), "`depth` is different from `len(dilated)`"
            depth = len(dilated)
        else:
            raise ValueError("Not support dilated={}".format(dilated))

        if type(norm) is bool:
            assert depth is not None, "Specify `depth`"
            norm = [norm] * depth
        elif type(norm) is list:
            if depth is not None:
                assert depth == len(norm), "`depth` is different from `len(norm)`"
            depth = len(norm)
        else:
            raise ValueError("Not support norm={}".format(norm))

        if type(nonlinear) is bool or type(nonlinear) is str:
            assert depth is not None, "Specify `depth`"
            nonlinear = [nonlinear] * depth
        elif type(nonlinear) is list:
            if depth is not None:
                assert depth == len(nonlinear), "`depth` is different from `len(nonlinear)`"
            depth = len(nonlinear)
        else:
            raise ValueError("Not support nonlinear={}".format(nonlinear))

        self.growth_rate = growth_rate
        self.depth = depth

        net = []
        _in_channels = in_channels - sum(growth_rate)

        for idx in range(depth):
            if idx == 0:
                _in_channels = in_channels
            else:
                _in_channels = growth_rate[idx - 1]
            _out_channels = sum(growth_rate[idx:])

            if dilated[idx]:
                dilation = 2**idx
            else:
                dilation = 1

            conv_block = ConvBlock2d(_in_channels, _out_channels, kernel_size=kernel_size, stride=1, dilation=dilation, norm=norm[idx], nonlinear=nonlinear[idx], eps=eps)
            net.append(conv_block)

        self.net = nn.Sequential(*net)

    def forward(self, input):
        """
        Args:
            input: (batch_size, in_channels, H, W)
        Returns:
            output: (batch_size, out_channels, H, W), where out_channels = growth_rate[-1].
        """
        growth_rate, depth = self.growth_rate, self.depth

        for idx in range(depth):
            if idx == 0:
                x = input
                x_residual = 0
            else:
                _in_channels = growth_rate[idx - 1]
                sections = [_in_channels, sum(growth_rate[idx:])]
                x, x_residual = torch.split(x_residual, sections, dim=1)

            x = self.net[idx](x)
            x_residual = x_residual + x

        output = x_residual

        return output

"""
    Quantizatioin
"""
class QuantizableD2Block(nn.Module):
    def __init__(self, in_channels, growth_rate, kernel_size, dilated=True, norm=True, nonlinear='relu', depth=None, eps=EPS):
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

        if type(growth_rate) is int:
            assert depth is not None, "Specify `depth`"
            growth_rate = [growth_rate] * depth
        elif type(growth_rate) is list:
            if depth is not None:
                assert depth == len(growth_rate), "`depth` is different from `len(growth_rate)`"
            depth = len(growth_rate)
        else:
            raise ValueError("Not support growth_rate={}".format(growth_rate))

        if type(dilated) is bool:
            assert depth is not None, "Specify `depth`"
            dilated = [dilated] * depth
        elif type(dilated) is list:
            if depth is not None:
                assert depth == len(dilated), "`depth` is different from `len(dilated)`"
            depth = len(dilated)
        else:
            raise ValueError("Not support dilated={}".format(dilated))

        if type(norm) is bool:
            assert depth is not None, "Specify `depth`"
            norm = [norm] * depth
        elif type(norm) is list:
            if depth is not None:
                assert depth == len(norm), "`depth` is different from `len(norm)`"
            depth = len(norm)
        else:
            raise ValueError("Not support norm={}".format(norm))

        if type(nonlinear) is bool or type(nonlinear) is str:
            assert depth is not None, "Specify `depth`"
            nonlinear = [nonlinear] * depth
        elif type(nonlinear) is list:
            if depth is not None:
                assert depth == len(nonlinear), "`depth` is different from `len(nonlinear)`"
            depth = len(nonlinear)
        else:
            raise ValueError("Not support nonlinear={}".format(nonlinear))

        self.growth_rate = growth_rate
        self.depth = depth
        self.float_ops = nnq.FloatFunctional()

        net = []
        _in_channels = in_channels - sum(growth_rate)

        for idx in range(depth):
            if idx == 0:
                _in_channels = in_channels
            else:
                _in_channels = growth_rate[idx - 1]
            _out_channels = sum(growth_rate[idx:])

            if dilated[idx]:
                dilation = 2**idx
            else:
                dilation = 1

            conv_block = QuantizableConvBlock2d(_in_channels, _out_channels, kernel_size=kernel_size, stride=1, dilation=dilation, norm=norm[idx], nonlinear=nonlinear[idx], eps=eps)
            net.append(conv_block)

        self.net = nn.Sequential(*net)

    def forward(self, input):
        """
        Args:
            input: (batch_size, in_channels, H, W)
        Returns:
            output: (batch_size, out_channels, H, W), where out_channels = growth_rate[-1].
        """
        growth_rate, depth = self.growth_rate, self.depth

        for idx in range(depth):
            if idx == 0:
                x = input
                x_residual = None
            else:
                _in_channels = growth_rate[idx - 1]
                sections = [_in_channels, sum(growth_rate[idx:])]
                x, x_residual = torch.split(x_residual, sections, dim=1)

            x = self.net[idx](x)

            if x_residual is None:
                x_residual = x
            else:
                x_residual = self.float_ops.add(x_residual, x)

        output = x_residual

        return output

def _test_d2block():
    batch_size = 4
    n_bins, n_frames = 16, 64
    in_channels = 3
    kernel_size = (3, 3)
    depth = 4

    input = torch.randn(batch_size, in_channels, n_bins, n_frames)

    print("-"*10, "D2 Block when `growth_rate` is given as int and `dilated` is given as bool.", "-"*10)

    growth_rate = 2
    dilated = True
    model = D2Block(in_channels, growth_rate, kernel_size=kernel_size, dilated=dilated, depth=depth)

    print("-"*10, "D2 Block", "-"*10)
    print(model)
    output = model(input)
    print(input.size(), output.size())
    print()

    print("-"*10, "D2 Block when `growth_rate` is given as list and `dilated` is given as bool.", "-"*10)

    growth_rate = [3, 4, 5, 6] # depth = 4
    dilated = False
    model = D2Block(in_channels, growth_rate, kernel_size=kernel_size, dilated=dilated)

    print(model)
    output = model(input)
    print(input.size(), output.size())
    print()

    print("-"*10, "D2 Block when `growth_rate` is given as list and `dilated` is given as list.", "-"*10)

    growth_rate = [3, 4, 5, 6] # depth = 4
    dilated = [True, False, False, True] # depth = 4
    model = D2Block(in_channels, growth_rate, kernel_size=kernel_size, dilated=dilated)

    print(model)
    output = model(input)
    print(input.size(), output.size())

if __name__ == '__main__':
    torch.manual_seed(111)

    print("="*10, "D2 Block", "="*10)
    _test_d2block()