import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils_tasnet import choose_layer_norm
from models.glu import GLU1d

EPS=1e-12

class FurcaNetBase(nn.Module):
    def __init__(self, num_conv_blocks, num_lstm_blocks, causal=False):
        super().__init__()
        
        self.num_conv_blocks, self.num_lstm_blocks = num_conv_blocks, num_lstm_blocks
        self.causal = causal
        num_directions = 2 # bi-direction
        
        self.gcn = GatedConvNet()
        self.stacked_lstm = nn.LSTM(num_features, hidden_channels//num_directions, layers=num_lstm_blocks, batch_first=True, bidirectional=True)
        self.dnn = nn.Linear()


class GatedConvNet(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=None, num_blocks=10, kernel_size=3, stride=1, nonlinear='sigmoid', norm=True, causal=False, eps=EPS):
        """
        Args:
            in_channels
            out_channels
            hidden_channels
            num_blocks
            kernel_size <int> or <list<int>>: 
            nonlinear <str>: nonlinear function of gated conv layer. Default: 'sigmoid'
            norm <bool>: normalization
            causal <bool>: causality
            eps

        Returns:

        """
        super().__init__()
        
        self.num_blocks = num_blocks

        if type(kernel_size) is int:
            kernel_size = [kernel_size] * num_blocks
        elif type(kernel_size) is list:
            if len(kernel_size) != num_blocks:
                raise ValueError("Invalid length of `kernel_size`")
        else:
            raise ValueError("Invalid type of `kernel_size`.")

        if type(stride) is int:
            stride = [stride] * num_blocks
        elif type(stride) is list:
            if len(stride) != num_blocks:
                raise ValueError("Invalid length of `stride`")
        else:
            raise ValueError("Invalid type of `stridee`.")
        
        net = []
        
        for idx in range(num_blocks):
            if idx == 0:
                net.append(GatedConvBlock(in_channels, hidden_channels, kernel_size=kernel_size[idx], stride=stride[idx], nonlinear=nonlinear, norm=norm, causal=causal, eps=eps))
            elif idx == num_blocks - 1:
                net.append(GatedConvBlock(hidden_channels, out_channels, kernel_size=kernel_size[idx], stride=stride[idx], nonlinear=nonlinear, norm=norm, causal=causal, eps=eps))
            else:
                net.append(GatedConvBlock(hidden_channels, hidden_channels, kernel_size=kernel_size[idx], stride=stride[idx], nonlinear=nonlinear, norm=norm, causal=causal, eps=eps))
        
        self.net = nn.Sequential(*net)
        
    def forward(self, input):
        output = self.net(input)
        
        return output

class GatedConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=80, stride=1, nonlinear='sigmoid', norm=True, causal=False, eps=EPS):
        super().__init__()

        self.kernel_size, self.stride = kernel_size, stride
        self.norm = norm
        self.eps = eps

        if nonlinear == 'sigmoid':
            self.gated_conv1d = GLU1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        else:
            raise ValueError("Not support nonlinear function {}.".format(nonlinear))
        
        if self.norm:
            self.norm1d = choose_layer_norm(out_channels, causal=causal, eps=eps)
    
    def forward(self, input):
        """
        Args:
            input (batch_size, in_channels, T)
        Returns:
            output (batch_size, out_channels, T'): T' = T // stride
        """
        kernel_size, stride = self.kernel_size, self.stride
        _, _, T = input.size()

        padding = kernel_size - stride
        padding_left = padding//2
        padding_right = padding - padding_left

        input = F.pad(input, (padding_left, padding_right))

        x = self.gated_conv1d(input)

        if self.norm:
            output = self.norm1d(x)
        else:
            output = x

        return output

def _test_gated_conv_block():
    torch.manual_seed(111)

    batch_size = 4
    in_channels, out_channels = 3, 16
    T = 128

    input = torch.rand(batch_size, in_channels, T, dtype=torch.float)

    gated_conv_block = GatedConvBlock(in_channels, out_channels)
    print(gated_conv_block)
    output = gated_conv_block(input)
    print(input.size(), output.size())

def _test_gated_conv_net():
    torch.manual_seed(111)

    batch_size = 4
    in_channels, out_channels, hidden_channels = 3, 16, 32
    T = 1024
    num_blocks = 10
    kernel_size = [1000] + [80] * (num_blocks - 1)

    input = torch.rand(batch_size, in_channels, T, dtype=torch.float)

    gated_conv_block = GatedConvNet(in_channels, out_channels, hidden_channels=hidden_channels, num_blocks=num_blocks, kernel_size=kernel_size)
    print(gated_conv_block)
    output = gated_conv_block(input)
    print(input.size(), output.size())

if __name__ == '__main__':
    print("="*10, "Gated Conv Block", "="*10)
    # _test_gated_conv_block()

    print("="*10, "Gated Conv Net", "="*10)
    _test_gated_conv_net()
