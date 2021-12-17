import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-12

"""
Reference: 
See 
"""
class DenseNet(nn.Module):
    def __init__(self, in_channels, num_features, growth_rate, kernel_size, stride=(1,1), hidden_channels=128, num_blocks=3, num_layers=[2,3,4], eps=EPS):
        super().__init__()

        self.num_blocks = num_blocks

        if type(num_layers) is list:
            assert num_blocks == len(num_layers), "`num_blocks` must be equal to `len(num_layers)`"
        else:
            num_layers = [
                num_layers for _ in range(num_blocks)
            ]

        self.preprocess = nn.Conv2d(in_channels, num_features, kernel_size=(1,1))

        net = []

        for _num_layers in num_layers:
            out_channels = (num_features + _num_layers * growth_rate)//2

            net.append(DenseBlock(num_features, out_channels, growth_rate, kernel_size, stride=stride, hidden_channels=hidden_channels, num_layers=_num_layers, eps=eps))

            num_features = out_channels

        self.net = nn.Sequential(*net)

    def forward(self, input):
        x = self.preprocess(input)
        for idx in range(self.num_blocks):
            x = self.net[idx](x)
        output = x

        return output

class DenseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, growth_rate, kernel_size, stride=(1,1), hidden_channels=128, num_layers=3, eps=EPS):
        super().__init__()

        net = []

        num_features = in_channels

        for idx in range(num_layers):
            net.append(DenseLayer(num_features, growth_rate, kernel_size, stride=stride, hidden_channels=hidden_channels, eps=eps))
            num_features += growth_rate

        self.net = nn.Sequential(*net)
        self.transition2d = Transition2d(num_features, out_channels, eps)

    def forward(self, input):
        x = self.net(input)
        output = self.transition2d(x)

        return output

class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, kernel_size, stride=(1,1), hidden_channels=128, eps=EPS):
        super().__init__()

        self.kernel_size, self.stride = kernel_size, stride

        self.norm2d1 = nn.BatchNorm2d(in_channels, eps=eps)
        self.relu1 = nn.ReLU()
        self.bottleneck_conv2d = nn.Conv2d(in_channels, hidden_channels, kernel_size=(1,1), stride=(1,1), bias=False)
        self.norm2d2 = nn.BatchNorm2d(hidden_channels, eps=eps)
        self.relu2 = nn.ReLU()
        self.conv2d = nn.Conv2d(hidden_channels, growth_rate, kernel_size=kernel_size, stride=stride, bias=False)

    def forward(self, input):
        _, _, H_in, W_in = input.size()
        Kh, Kw = self.kernel_size
        Sh, Sw = self.stride

        padding_height = H_in * (Sh - 1) + Kh  - Sh
        padding_width = W_in * (Sw - 1) + Kw  - Sw
        padding_up = padding_height // 2
        padding_bottom = padding_height - padding_up
        padding_left = padding_width // 2
        padding_right = padding_width - padding_left

        x = self.norm2d1(input)
        x = self.relu1(x)
        x = self.bottleneck_conv2d(x)
        x = self.norm2d2(x)
        x = self.relu2(x)
        x = F.pad(x, (padding_left, padding_right, padding_up, padding_bottom))
        x = self.conv2d(x)
        output = torch.cat([input, x], dim=1)

        return output

class Transition2d(nn.Module):
    def __init__(self, in_channels, out_channels, eps=EPS):
        super().__init__()

        self.norm2d = nn.BatchNorm2d(in_channels, eps=eps)
        self.relu = nn.ReLU()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=(1,1), stride=(1,1), bias=False)
        self.pool2d = nn.AvgPool2d(kernel_size=(2,2), stride=(2,2))

    def forward(self, input):
        x = self.norm2d(input)
        x = self.relu(x)
        x = self.conv2d(x)
        output = self.pool2d(x)

        return output

def _test_dense_block():
    torch.manual_seed(111)

    batch_size = 4
    H, W = 16, 32
    in_channels, out_channels, growth_rate = 4, 32, 16

    input = torch.randn(batch_size, in_channels, H, W)

    model = DenseBlock(in_channels, out_channels, growth_rate=growth_rate, kernel_size=(3, 5))
    print(model)
    output = model(input)

    print(input.size(), output.size())

def _test_densenet():
    torch.manual_seed(111)

    batch_size = 4
    H, W = 16, 32
    in_channels, num_features, growth_rate = 3, 4, 8

    input = torch.randn(batch_size, in_channels, H, W)

    model = DenseNet(in_channels, num_features, growth_rate=growth_rate, kernel_size=(3, 5))
    print(model)
    output = model(input)

    print(input.size(), output.size())

def _test_densenet_official():
    model = torch.hub.load('pytorch/vision:v0.6.0', 'densenet121', pretrained=True)
    print(model)


if __name__ == '__main__':
    _test_dense_block()
    print()

    _test_densenet()
    print()
    # _test_densenet_official()