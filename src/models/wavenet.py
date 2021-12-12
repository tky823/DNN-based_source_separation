import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.model import choose_nonlinear
from utils.tasnet import choose_layer_norm
from conv import DepthwiseSeparableConv1d

EPS = 1e-12

class WaveNet(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=256, skip_channels=256, kernel_size=3, num_blocks=3, num_layers=10, dilated=True, separable=False, causal=True, nonlinear='gated', norm=True, output_nonlinear=None, conditioning=None, enc_dim=None, enc_kernel_size=None, enc_stride=None, eps=EPS):
        super().__init__()

        self.num_blocks = num_blocks

        self.causal_conv1d = nn.Conv1d(in_channels, hidden_channels, kernel_size=1, stride=1, bias=False)

        net = []

        for idx in range(num_blocks):
            net.append(ConvBlock1d(hidden_channels, skip_channels, kernel_size=kernel_size, num_layers=num_layers, dilated=dilated, separable=separable, causal=causal, nonlinear=nonlinear, norm=norm, conditioning=conditioning, enc_dim=enc_dim, enc_kernel_size=enc_kernel_size, enc_stride=enc_stride, eps=eps))

        self.net = nn.Sequential(*net)
        end_net = []

        end_net.append(nn.ReLU())
        end_net.append(nn.Conv1d(skip_channels, hidden_channels, kernel_size=1, stride=1, bias=False))
        end_net.append(nn.ReLU())
        end_net.append(nn.Conv1d(hidden_channels, out_channels, kernel_size=1, stride=1, bias=False))

        if output_nonlinear is not None:
            if output_nonlinear == 'softmax':
                kwargs = {
                    "dim": 1
                }
            else:
                kwargs = {}

            module = choose_nonlinear(output_nonlinear, **kwargs)
            end_net.append(module)

        self.end_net = nn.Sequential(*end_net)

    def forward(self, input, enc_h=None):
        num_blocks = self.num_blocks

        x = self.causal_conv1d(input)
        skip_connection = 0

        for idx in range(num_blocks):
            x, skip = self.net[idx](x, enc_h=enc_h)
            skip_connection = skip_connection + skip

        output = self.end_net(skip_connection)

        return output

    @classmethod
    def build_model(cls, model_path, load_state_dict=False):
        config = torch.load(model_path, map_location=lambda storage, loc: storage)

        model = cls(in_channels=config['in_channels'], out_channels=config['out_channels'], hidden_channels=config['hidden_channels'], skip_channels=config['skip_channels'], kernel_size=config['kernel_size'], num_blocks=config['num_blocks'], num_layers=config['num_layers'], dilated=config['dilated'], separable=config['separable'], causal=config['causal'], nonlinear=config['nonlinear'], norm=config['norm'], output_nonlinear=config['output_nonlinear'], conditioning=config['conditioning'], enc_dim=config['enc_dim'], enc_kernel_size=config['enc_kernel_size'], enc_stride=config['enc_stride'])

        if load_state_dict:
            model.load_state_dict(config['state_dict'])

        return model

    def get_config(self):
        config = {
            'in_channels': self.in_channels, 'out_channels': self.out_channels, 'hidden_channels': self.hidden_channels, 'skip_channels': self.skip_channels,
            'kernel_size': self.kernel_size,
            'num_blocks': self.num_blocks, 'num_layers': self.num_layers,
            'dilated': self.dilated, 'separable': self.separable,
            'causal': self.causal,
            'nonlinear': self.nonlinear, 'norm': self.norm,
            'output_nonlinear': self.output_nonlinear,
            'conditioning': self.conditioning,
            'enc_dim': self.enc_dim,
            'enc_kernel_size': self.enc_kernel_size, 'enc_stride': self.enc_stride
        }

        return config

    @property
    def num_parameters(self):
        _num_parameters = 0

        for p in self.parameters():
            if p.requires_grad:
                _num_parameters += p.numel()

        return _num_parameters

class ConvBlock1d(nn.Module):
    def __init__(self, hidden_channels, skip_channels, kernel_size=3, num_layers=10, dilated=True, separable=False, causal=True, nonlinear='gated', norm=True, conditioning=None, enc_dim=None, enc_kernel_size=None, enc_stride=None, eps=EPS):
        super().__init__()

        self.num_layers = num_layers

        net = []

        for idx in range(num_layers):
            if dilated:
                dilation = 2**idx
                stride = 1
            else:
                dilation = 1
                stride = 2
            net.append(ResidualBlock1d(hidden_channels, hidden_channels, skip_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, separable=separable, causal=causal, nonlinear=nonlinear, norm=norm, conditioning=conditioning, enc_dim=enc_dim, enc_kernel_size=enc_kernel_size, enc_stride=enc_stride, eps=eps))

        self.net = nn.Sequential(*net)

    def forward(self, input, enc_h=None):
        num_layers = self.num_layers

        x = input
        skip_connection = 0

        for idx in range(num_layers):
            x, skip = self.net[idx](x, enc_h=enc_h)
            skip_connection = skip_connection + skip

        return x, skip_connection

class ResidualBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels, kernel_size=3, stride=2, dilation=1, separable=False, causal=True, nonlinear='gated', norm=True, conditioning=None, enc_dim=None, enc_kernel_size=None, enc_stride=None, eps=EPS):
        super().__init__()
        
        self.kernel_size, self.stride, self.dilation = kernel_size, stride, dilation
        self.separable = separable
        self.nonlinear, self.norm = nonlinear, norm

        # TODO: implement nonlinear & norm
        if nonlinear == 'gated':
            self.conv1d = GatedConv1d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, separable=separable, causal=causal, conditioning=conditioning, enc_dim=enc_dim, enc_kernel_size=enc_kernel_size, enc_stride=enc_stride)
        elif nonlinear == 'relu':
            raise NotImplementedError("No implementation of nonlinear ReLU.")
            net = []
            if separable:
                net.append(SeparableConv1d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, bias=False))
            else:
                net.append(nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, bias=False))
            net.append(nn.ReLU())
            self.conv1d = nn.Sequential(*net)
        else:
            raise ValueError("Not support {}".format(nonlinear))

        if norm:
            if causal:
                self.norm1d = choose_layer_norm(out_channels, causal, eps=eps)
            else:
                self.norm1d = nn.BatchNorm1d(out_channels, eps=eps)

        self.bottleneck_conv1d_output = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1)
        self.bottleneck_conv1d_skip = nn.Conv1d(in_channels, skip_channels, kernel_size=1, stride=1)

    def forward(self, input, enc_h=None):
        norm = self.norm

        residual = input
        x = self.conv1d(input, enc_h=enc_h)
        if norm:
            x = self.norm1d(x)
        output = self.bottleneck_conv1d_output(x)
        skip = self.bottleneck_conv1d_skip(x)
        output = output + residual

        return output, skip

class GatedConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, dilation=1, separable=False, causal=True, conditioning=None, enc_dim=None, enc_kernel_size=None, enc_stride=None):
        super().__init__()

        if conditioning is not None:
            if conditioning == 'global':
                if enc_kernel_size is not None or enc_stride is not None:
                    raise ValueError("enc_kernel_size and enc_stride must be NOT None")
            elif conditioning == 'local':
                if enc_kernel_size is None or enc_stride is None:
                    raise ValueError("enc_kernel_size and enc_stride must be None")
                self.enc_kernel_size, self.enc_stride = enc_kernel_size, enc_stride
            else:
                raise ValueError("Not support {}".format(conditioning))
            if enc_dim is None:
                raise ValueError("enc_dim must be NOT None")

            self.conditioning = conditioning
        else:
            if enc_dim is not None:
                raise ValueError("enc_dim must be None")
            if enc_kernel_size is not None or enc_stride is not None:
                raise ValueError("enc_kernel_size and enc_stride must be None")

            self.conditioning = None

        self.kernel_size, self.stride, self.dilation = kernel_size, stride, dilation
        self.causal = causal

        if separable:
            self.tanh_conv1d = DepthwiseSeparableConv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, bias=False)
        else:
            self.tanh_conv1d = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, bias=False)

        if conditioning is not None:
            if conditioning == 'global':
                self.embed_tanh_linear = nn.Linear(enc_dim, out_channels)
            elif conditioning=='local':
                self.embed_tanh_map = nn.ConvTranspose1d(enc_dim, enc_dim, kernel_size=enc_kernel_size, stride=enc_stride, bias=False)
                self.embed_tanh_conv1d = nn.Conv1d(enc_dim, out_channels, kernel_size=1, stride=1, bias=False)
            else:
                raise NotImplementedError("Choose 'global' or 'local' for conditioning.")

        self.tanh = nn.Tanh()

        if separable:
            self.sigmoid_conv1d = DepthwiseSeparableConv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, bias=False)
        else:
            self.sigmoid_conv1d = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, bias=False)

        if conditioning is not None:
            if conditioning == 'global':
                self.embed_sigmoid_linear = nn.Linear(enc_dim, out_channels)
            elif conditioning=='local':
                self.embed_sigmoid_map = nn.ConvTranspose1d(enc_dim, enc_dim, kernel_size=enc_kernel_size, stride=enc_stride, bias=False)
                self.embed_sigmoid_conv1d = nn.Conv1d(enc_dim, out_channels, kernel_size=1, stride=1, bias=False)
            else:
                raise NotImplementedError("Choose 'global' or 'local' for conditioning.")

        self.sigmoid = nn.Sigmoid()

    def forward(self, input, enc_h=None):
        kernel_size, stride, dilation = self.kernel_size, self.stride, self.dilation
        causal = self.causal
        conditioning = self.conditioning

        _, _, T = input.size()

        padding = (T - 1) * stride + (kernel_size - 1) * dilation + 1 - T

        if causal:
            padding_left = padding
            padding_right = 0
        else:
            padding_left = padding // 2
            padding_right = padding - padding_left

        x = F.pad(input, (padding_left, padding_right))

        if conditioning is None:
            if enc_h is not None:
                raise ValueError("Set enc_h to be 'None'.")
        elif conditioning == 'local':
            enc_kernel_size, enc_stride = self.enc_kernel_size, self.enc_stride

        x_tanh = self.tanh_conv1d(x)

        if conditioning is not None:
            if conditioning == 'global':
                y_tanh = self.embed_tanh_linear(enc_h)
                y_tanh = y_tanh.unsqueeze(dim=2)
                x_tanh =  x_tanh + y_tanh
            elif conditioning == 'local':
                y_tanh = self.embed_tanh_map(enc_h)
                padding = enc_kernel_size - enc_stride
                if causal:
                    padding_left = 0
                    padding_right = padding
                else:
                    padding_left = padding // 2
                    padding_right = padding - padding_left

                y_tanh = F.pad(y_tanh, (-padding_left, -padding_right))
                x_tanh = x_tanh + self.embed_tanh_conv1d(y_tanh)
            else:
                raise NotImplementedError("Choose 'global' or 'local' for conditioning.")

        x_tanh = self.tanh(x_tanh)
        x_sigmoid = self.sigmoid_conv1d(x)

        if conditioning is not None:
            if conditioning == 'global':
                y_sigmoid = self.embed_sigmoid_linear(enc_h)
                y_sigmoid = y_sigmoid.unsqueeze(dim=2)
                x_sigmoid =  x_sigmoid + y_sigmoid
            elif conditioning == 'local':
                y_sigmoid = self.embed_sigmoid_map(enc_h)
                y_sigmoid = F.pad(y_sigmoid, (-padding_left, -padding_right))
                x_sigmoid =  x_sigmoid + self.embed_sigmoid_conv1d(y_sigmoid)
            else:
                raise NotImplementedError("Choose 'global' or 'local' for conditioning.")

        x_sigmoid = self.sigmoid(x_sigmoid)
        output = x_tanh * x_sigmoid

        return output

def _test_wavenet():
    batch_size, T = 4, 1024
    hidden_channels, skip_channels = 128, 256
    kernel_size = 2
    num_blocks, num_layers = 2, 3
    nonlinear = 'gated'

    # Example 1: w/o conditioning
    print('-'*10, "Example 1: w/o conditioning", '-'*10)

    in_channels, out_channels = 1, 1
    dilated, separable, causal = False, False, True
    norm = True
    output_nonlinear = 'tanh'

    model = WaveNet(in_channels=in_channels, out_channels=out_channels, hidden_channels=hidden_channels, skip_channels=skip_channels, kernel_size=kernel_size, num_blocks=num_blocks, num_layers=num_layers, dilated=dilated, separable=separable, causal=causal, nonlinear=nonlinear, norm=norm, output_nonlinear=output_nonlinear)
    print(model)
    print("# Parameters: {}".format(model.num_parameters))

    input = torch.randint(0, 10, (batch_size, in_channels, T), dtype=torch.float)
    output = model(input)
    print(input.size(), output.size())
    print()

    # Example 2: global conditioning
    print('-'*10, "Example 2: global conditioning", '-'*10)
    in_channels, out_channels = 128, 128
    dilated, separable, causal = True, True, False
    norm = True
    output_nonlinear = 'softmax'
    conditioning = 'global'
    enc_dim = 4

    model = WaveNet(in_channels=in_channels, out_channels=out_channels, hidden_channels=hidden_channels, skip_channels=skip_channels, kernel_size=kernel_size, num_blocks=num_blocks, num_layers=num_layers, dilated=dilated, separable=separable, causal=causal, nonlinear=nonlinear, norm=norm, output_nonlinear=output_nonlinear, conditioning=conditioning, enc_dim=enc_dim)
    print(model)
    print("# Parameters: {}".format(model.num_parameters))

    input = torch.randint(0, 10, (batch_size, in_channels, T), dtype=torch.float)
    enc_h = torch.randint(0, 10, (batch_size, enc_dim), dtype=torch.float)
    output = model(input, enc_h=enc_h)
    print(input.size(), enc_h.size(), output.size())
    print()

    # Example 3: w/ local conditioning
    print('-'*10, "Example 3: w/ local conditioning", '-'*10)

    in_channels, out_channels = 1, 2
    dilated, separable, causal = True, False, False
    norm = False
    output_nonlinear = None
    conditioning = 'local'
    enc_dim, enc_T = 2, 512
    enc_kernel_size, enc_stride = 3, T // enc_T

    model = WaveNet(in_channels=in_channels, out_channels=out_channels, hidden_channels=hidden_channels, skip_channels=skip_channels, kernel_size=kernel_size, num_blocks=num_blocks, num_layers=num_layers, dilated=dilated, separable=separable, causal=causal, nonlinear=nonlinear, norm=norm, output_nonlinear=output_nonlinear, conditioning=conditioning, enc_dim=enc_dim, enc_kernel_size=enc_kernel_size, enc_stride=enc_stride)
    print(model)
    print("# Parameters: {}".format(model.num_parameters))

    input = torch.randint(0, 10, (batch_size, in_channels, T), dtype=torch.float)
    enc_h = torch.randint(0, 10, (batch_size, enc_dim, enc_T), dtype=torch.float)
    output = model(input, enc_h=enc_h)
    print(input.size(), enc_h.size(), output.size())

if __name__ == '__main__':
    # You have to export python path like: export PYTHONPATH="../:$PYTHONPATH"
    print('='*10, "Wave Net", '='*10)
    _test_wavenet()