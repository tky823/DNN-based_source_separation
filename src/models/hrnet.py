import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.model import choose_nonlinear
from models.resnet import ResidualBlock2d

EPS = 1e-12

class HRNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, bottleneck_channels, kernel_size=(3,3), scale=(2,2), upsample='bilinear', downsample='conv', nonlinear='relu', mask_nonlinear='relu', num_stacks=1, in_num_stacks=2, out_num_stacks=2, eps=EPS):
        super().__init__()

        if type(num_stacks) is int:
            num_stacks = [num_stacks] * len(hidden_channels)
        else:
            assert len(num_stacks) == len(hidden_channels), "Invalid length of num_stacks."

        self.conv2d_in = StackedResidualBlock2d(in_channels, hidden_channels[0], bottleneck_channels=bottleneck_channels, kernel_size=kernel_size, nonlinear=nonlinear, num_stacks=in_num_stacks, eps=eps)
        self.backbone = HRNetBackbone(hidden_channels, bottleneck_channels, kernel_size=kernel_size, scale=scale, upsample=upsample, downsample=downsample, nonlinear=nonlinear, num_stacks=num_stacks, eps=eps)
        self.conv2d_out = StackedResidualBlock2d(sum(hidden_channels), in_channels, bottleneck_channels=bottleneck_channels, kernel_size=kernel_size, nonlinear=nonlinear, num_stacks=out_num_stacks, eps=eps)
        self.mask_nonlinear2d = choose_nonlinear(mask_nonlinear)

        self.in_channels, self.hidden_channels, self.bottleneck_channels = in_channels, hidden_channels, bottleneck_channels
        self.kernel_size = kernel_size
        self.scale = scale
        self.upsample, self.downsample = upsample, downsample
        self.nonlinear, self.mask_nonlinear = nonlinear, mask_nonlinear
        self.num_stacks = num_stacks
        self.in_num_stacks, self.out_num_stacks = in_num_stacks, out_num_stacks

        self.eps = eps

    def forward(self, input):
        mask = self.estimate_mask(input)
        output = mask * input

        return output

    def estimate_mask(self, input):
        x = self.conv2d_in(input)
        x = self.backbone(x)
        x = self.conv2d_out(x)
        mask = self.mask_nonlinear2d(x)

        return mask

    def get_config(self):
        in_channels = self.in_channels
        hidden_channels, bottleneck_channels = self.hidden_channels, self.bottleneck_channels
        kernel_size = self.kernel_size
        scale = self.scale
        upsample, downsample = self.upsample, self.downsample
        nonlinear, mask_nonlinear = self.nonlinear, self.mask_nonlinear
        num_stacks = self.num_stacks
        in_num_stacks, out_num_stacks = self.in_num_stacks, self.out_num_stacks

        eps = self.eps

        config = {
            'in_channels': in_channels,
            'hidden_channels': hidden_channels, 'bottleneck_channels': bottleneck_channels,
            'kernel_size': kernel_size,
            'scale': scale,
            'upsample': upsample, 'downsample': downsample,
            'nonlinear': nonlinear, 'mask_nonlinear': mask_nonlinear,
            'num_stacks': num_stacks,
            'in_num_stacks': in_num_stacks, 'out_num_stacks': out_num_stacks,
            'eps': eps
        }

        return config

    @classmethod
    def build_model(cls, model_path, load_state_dict=False):
        config = torch.load(model_path, map_location=lambda storage, loc: storage)

        in_channels = config['in_channels']
        hidden_channels, bottleneck_channels = config['hidden_channels'], config['bottleneck_channels']
        kernel_size = config['kernel_size']
        scale = config['scale']
        upsample, downsample = config['upsample'], config['downsample']
        nonlinear, mask_nonlinear = config['nonlinear'], config['mask_nonlinear']
        num_stacks = config['num_stacks']
        in_num_stacks, out_num_stacks = config['in_num_stacks'], config['out_num_stacks']

        eps = config['eps']

        model = cls(
            in_channels,
            hidden_channels=hidden_channels, bottleneck_channels=bottleneck_channels,
            kernel_size=kernel_size,
            scale=scale,
            upsample=upsample, downsample=downsample,
            nonlinear=nonlinear, mask_nonlinear=mask_nonlinear,
            num_stacks=num_stacks,
            in_num_stacks=in_num_stacks, out_num_stacks=out_num_stacks,
            eps=eps
        )

        if load_state_dict:
            model.load_state_dict(config['state_dict'])

        return model

    @classmethod
    def build_from_config(cls, config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        in_channels = config['in_channels']
        hidden_channels, bottleneck_channels = config['hidden_channels'], config['bottleneck_channels']
        kernel_size = config['kernel_size']
        scale = config['scale']
        upsample, downsample = config['upsample'], config['downsample']
        nonlinear, mask_nonlinear = config['nonlinear'], config['mask_nonlinear']
        num_stacks = config['num_stacks']
        in_num_stacks, out_num_stacks = config['in_num_stacks'], config['out_num_stacks']

        eps = config.get('eps') or EPS

        model = cls(
            in_channels,
            hidden_channels=hidden_channels, bottleneck_channels=bottleneck_channels,
            kernel_size=kernel_size,
            scale=scale,
            upsample=upsample, downsample=downsample,
            nonlinear=nonlinear, mask_nonlinear=mask_nonlinear,
            num_stacks=num_stacks,
            in_num_stacks=in_num_stacks, out_num_stacks=out_num_stacks,
            eps=eps
        )

        return model

    @property
    def num_parameters(self):
        _num_parameters = 0

        for p in self.parameters():
            if p.requires_grad:
                _num_parameters += p.numel()

        return _num_parameters

class HRNetBackbone(nn.Module):
    def __init__(self, hidden_channels, bottleneck_channels, kernel_size=(3,3), scale=(2,2), upsample='bilinear', downsample='conv', nonlinear='relu', num_stacks=1, eps=EPS):
        super().__init__()

        num_stages = len(hidden_channels)

        if type(num_stacks) is int:
            num_stacks = [num_stacks] * num_stages
        else:
            assert len(num_stacks) == num_stages, "Inavalid length of num_stacks."

        net = []

        for idx in range(num_stages):
            if idx == num_stages - 1:
                block = StackedParallelResidualBlock2d(hidden_channels[:idx + 1], 0, bottleneck_channels=bottleneck_channels, kernel_size=kernel_size, scale=scale, upsample=upsample, downsample=downsample, nonlinear=nonlinear, num_stacks=num_stacks[idx], eps=eps)
            else:
                block = StackedParallelResidualBlock2d(hidden_channels[:idx + 1], hidden_channels[idx + 1], bottleneck_channels=bottleneck_channels, kernel_size=kernel_size, scale=scale, upsample=upsample, downsample=downsample, nonlinear=nonlinear, num_stacks=num_stacks[idx], eps=eps)

            net.append(block)

        self.net = nn.Sequential(*net)
        self.concat_mix_block2d = ConcatMixBlock2d(hidden_channels, scale=scale, upsample=upsample, eps=eps)

        self.num_stages = num_stages

    def forward(self, input):
        x = [input]
        for idx in range(self.num_stages):
            x = self.net[idx](x)

        output = self.concat_mix_block2d(x)

        return output

class StackedParallelResidualBlock2d(nn.Module):
    def __init__(self, in_channels, additional_channels, bottleneck_channels, kernel_size=(3,3), scale=(2,2), upsample='bilinear', downsample='conv', nonlinear='relu', num_stacks=1, eps=EPS):
        super().__init__()

        self.num_stacks = num_stacks
        self.max_level = len(in_channels) - 1

        residual_block2d = []

        for idx in range(num_stacks):
            blocks = []
            for _in_channels in in_channels:
                block = ResidualBlock2d(_in_channels, _in_channels, bottleneck_channels, kernel_size=kernel_size, nonlinear=nonlinear, eps=eps)
                blocks.append(block)
            blocks = nn.ModuleList(blocks)
            residual_block2d.append(blocks)

        self.residual_block2d = nn.ModuleList(residual_block2d)

        self.mix_block2d = MixBlock2d(in_channels, additional_channels, scale=scale, upsample=upsample, downsample=downsample, eps=eps)

    def forward(self, input):
        x_in = input

        for stack_idx in range(self.num_stacks):
            x_out = []
            module = self.residual_block2d[stack_idx]
            for level_idx in range(self.max_level + 1):
                x = module[level_idx](x_in[level_idx])
                x_out.append(x)
            x_in = x_out

        output = self.mix_block2d(x_out)

        return output

class StackedResidualBlock2d(nn.Module):
    def __init__(self, in_channels, out_channels, bottleneck_channels, kernel_size=(3,3), nonlinear='relu', num_stacks=1, eps=EPS):
        super().__init__()

        self.num_stacks = num_stacks
        net = []

        for idx in range(num_stacks):
            if idx == 0:
                block = ResidualBlock2d(in_channels, out_channels, bottleneck_channels, kernel_size=kernel_size, nonlinear=nonlinear, eps=eps)
            else:
                block = ResidualBlock2d(out_channels, out_channels, bottleneck_channels, kernel_size=kernel_size, nonlinear=nonlinear, eps=eps)
            net.append(block)

        self.net = nn.ModuleList(net)

    def forward(self, input):
        x = input

        for idx in range(self.num_stacks):
            x = self.net[idx](x)

        output = x

        return output

class MixBlock2d(nn.Module):
    def __init__(self, in_channels, additional_channels, scale=(2,2), upsample='bilinear', downsample='conv', eps=EPS):
        super().__init__()

        max_level_in = len(in_channels) - 1
        blocks = []

        if additional_channels > 0:
            out_channels = in_channels + [additional_channels]
            max_level_out = max_level_in + 1
        else:
            out_channels = in_channels
            max_level_out = max_level_in

        for idx_out in range(max_level_out + 1):
            interpolate_block = []
            _out_channels = out_channels[idx_out]

            for idx_in in range(max_level_in + 1):
                _in_channels = in_channels[idx_in]

                sH, sW = scale
                sH, sW = sH**(idx_out - idx_in), sW**(idx_out - idx_in)
                _scale = (sH, sW)

                if idx_in < idx_out:
                    block = DownsampleBlock2d(_in_channels, _out_channels, scale=_scale, mode=downsample, eps=eps)
                elif idx_in > idx_out:
                    block = UpsampleBlock2d(_in_channels, _out_channels, scale=_scale, mode=upsample, eps=eps)
                else:
                    block = nn.Identity()

                interpolate_block.append(block)

            interpolate_block = nn.ModuleList(interpolate_block)
            blocks.append(interpolate_block)

        self.blocks = nn.ModuleList(blocks)

        self.max_level_in, self.max_level_out = max_level_in, max_level_out

    def forward(self, input):
        output = []

        for idx_out in range(self.max_level_out + 1):
            x_level_out = 0
            module = self.blocks[idx_out]

            for idx_in in range(self.max_level_in + 1):
                x = module[idx_in](input[idx_in])
                if type(x_level_out) is not int:
                    _, _, H_in, W_in = x.size()
                    _, _, H, W = x_level_out.size()
                    Ph, Pw = H_in - H, W_in - W
                    padding_top, padding_left = Ph // 2, Pw // 2
                    padding_bottom, padding_right = Ph - padding_top, Pw - padding_left
                    x = F.pad(x, (-padding_left, -padding_right, -padding_top, -padding_bottom))

                x_level_out = x_level_out + x

            output.append(x_level_out)

        return output

class ConcatMixBlock2d(nn.Module):
    def __init__(self, in_channels, scale=(2,2), upsample='bilinear', eps=EPS):
        super().__init__()

        max_level_in = len(in_channels) - 1
        net = []

        for idx_in in range(max_level_in + 1):
            _in_channels = in_channels[idx_in]

            sH, sW = scale
            sH, sW = 1 / (sH**idx_in), 1 / (sW**idx_in)
            _scale = (sH, sW)

            if idx_in == 0:
                block = nn.Identity()
            else:
                block = UpsampleBlock2d(_in_channels, _in_channels, scale=_scale, mode=upsample, eps=eps)

            net.append(block)

        self.net = nn.ModuleList(net)
        self.max_level_in = max_level_in

    def forward(self, input):
        output = []

        for idx in range(self.max_level_in + 1):
            x = self.net[idx](input[idx])

            if idx == 0:
                _, _, H, W = x.size()
            else:
                _, _, H_in, W_in = x.size()
                Ph, Pw = H_in - H, W_in - W
                padding_top, padding_left = Ph // 2, Pw // 2
                padding_bottom, padding_right = Ph - padding_top, Pw - padding_left
                x = F.pad(x, (-padding_left, -padding_right, -padding_top, -padding_bottom))

            output.append(x)

        output = torch.cat(output, dim=1)

        return output

class DownsampleBlock2d(nn.Module):
    def __init__(self, in_channels, out_channels, scale=(2,2), mode='conv', nonlinear='relu', eps=EPS):
        super().__init__()

        self.pointwise_conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=(1,1), stride=(1,1), bias=False)
        self.norm2d = nn.BatchNorm2d(out_channels, eps=eps)

        if mode != 'conv':
            raise NotImplementedError("Invalid upsample mode.")

        self.conv2d = nn.Conv2d(out_channels, out_channels, kernel_size=(3,3), stride=scale)
        self.nonlinear2d = choose_nonlinear(nonlinear)

        self.scale = scale
        self.mode = mode

    def forward(self, input):
        padding_hight, padding_width = 3 - 1, 3 - 1
        padding_top, padding_left = padding_hight // 2, padding_width // 2
        padding_bottom, padding_right = padding_hight - padding_top, padding_width - padding_left

        x = self.pointwise_conv2d(input)
        x = self.norm2d(x)
        x = F.pad(x, (padding_left, padding_right, padding_top, padding_bottom))
        x = self.conv2d(x)
        output = self.nonlinear2d(x)

        return output

class UpsampleBlock2d(nn.Module):
    def __init__(self, in_channels, out_channels, scale=(2,2), mode='bilinear', eps=EPS):
        super().__init__()

        self.pointwise_conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=(1,1), stride=(1,1), bias=False)
        self.norm2d = nn.BatchNorm2d(out_channels, eps=eps)

        if mode != 'bilinear':
            raise NotImplementedError("Invalid upsample mode.")

        self.scale = (1 / scale[0], 1 / scale[1])
        self.mode = mode

    def forward(self, input):
        x = self.pointwise_conv2d(input)
        x = self.norm2d(x)
        output = F.interpolate(x, scale_factor=self.scale, mode=self.mode, align_corners=False)

        return output

def _test_mix_block():
    batch_size = 4
    in_channels = [3, 5, 7]
    additional_channels = 0
    H, W = 32, 64

    input = [
        torch.randn(batch_size, c, H // (2**idx), W // (2**idx)) for idx, c in enumerate(in_channels)
    ]
    model = MixBlock2d(in_channels, additional_channels)
    output = model(input)

    print(model)

    for _input, _output in zip(input, output):
        print(_input.size(), _output.size())

    print()

    additional_channels = 9

    input = [
        torch.randn(batch_size, c, H // (2**idx), W // (2**idx)) for idx, c in enumerate(in_channels)
    ]
    model = MixBlock2d(in_channels, additional_channels)
    output = model(input)

    print(model)

    for _input, _output in zip(input, output):
        print(_input.size(), _output.size())

    print(output[-1].size())

def _test_stacked_parallel_residual_block():
    batch_size = 4
    H, W = 32, 64
    in_channels, hidden_channels = [3, 5, 7], 4
    kernel_size = (3, 3)

    input = [
        torch.randn(batch_size, c, H // (2**idx), W // (2**idx)) for idx, c in enumerate(in_channels)
    ]

    num_stacks = 1
    additional_channels = 0
    model = StackedParallelResidualBlock2d(in_channels, additional_channels, hidden_channels, kernel_size=kernel_size, num_stacks=num_stacks)
    output = model(input)

    print(model)
    for _input, _output in zip(input, output):
        print(_input.size(), _output.size())

    print()

    num_stacks = 1
    additional_channels = 9
    model = StackedParallelResidualBlock2d(in_channels, additional_channels, hidden_channels, kernel_size=kernel_size, num_stacks=num_stacks)
    output = model(input)

    print(model)
    for _input, _output in zip(input, output):
        print(_input.size(), _output.size())
    print(output[-1].size())

    print()

    num_stacks = 2
    additional_channels = 9
    model = StackedParallelResidualBlock2d(in_channels, additional_channels, hidden_channels, kernel_size=kernel_size, num_stacks=num_stacks)
    output = model(input)

    print(model)
    for _input, _output in zip(input, output):
        print(_input.size(), _output.size())
    print(output[-1].size())

def _test_hrnet_backbone():
    batch_size = 4
    H, W = 32, 65
    hidden_channels = [3, 5, 7]
    bottleneck_channels = 4
    kernel_size = (3, 3)

    input = torch.randn(batch_size, hidden_channels[0], H, W)

    model = HRNetBackbone(hidden_channels, bottleneck_channels, kernel_size=kernel_size)
    output = model(input)

    print(output.size())

def _test_hrnet():
    batch_size = 4
    H, W = 32, 66
    in_channels, hidden_channels = 2, [3, 5, 7]
    bottleneck_channels = 4
    kernel_size = (3, 3)

    input = torch.randn(batch_size, in_channels, H, W)

    model = HRNet(in_channels, hidden_channels, bottleneck_channels, kernel_size=kernel_size)
    output = model(input)

    print(output.size())

def _test_hrnet_paper():
    batch_size = 6
    H, W = 64, 129
    in_channels, hidden_channels = 2, [3, 5, 7, 9]
    bottleneck_channels = 4
    kernel_size = (3, 3)
    num_stacks = [1, 1, 4, 3]

    input = torch.randn(batch_size, in_channels, H, W)
    input = torch.abs(input)

    model = HRNet(in_channels, hidden_channels, bottleneck_channels, kernel_size=kernel_size, num_stacks=num_stacks)
    output = model(input)

    print(output.size())

def _test_hrnet_from_config():
    batch_size = 6
    H, W = 64, 129
    in_channels = 2

    input = torch.randn(batch_size, in_channels, H, W)
    input = torch.abs(input)

    model = HRNet.build_from_config("./data/hrnet/baseline.yaml")
    output = model(input)

    print(output.size())

if __name__ == '__main__':
    torch.manual_seed(111)

    print("="*10, "MixBlock2d", "="*10)
    _test_mix_block()
    print()

    print("="*10, "StackedParallelResidualBlock", "="*10)
    _test_stacked_parallel_residual_block()
    print()

    print("="*10, "HRNet backbone", "="*10)
    _test_hrnet_backbone()
    print()

    print("="*10, "HRNet", "="*10)
    _test_hrnet()
    print()

    print("="*10, "HRNet (paper)", "="*10)
    _test_hrnet_paper()
    print()

    print("="*10, "HRNet (from config)", "="*10)
    _test_hrnet_from_config()