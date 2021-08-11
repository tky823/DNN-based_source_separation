import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-12

class MetaTasNet(nn.Module):
    def __init__(self):
        pass

class Conv1dGenerated(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, embed_dim=None, bottleneck_channels=None, n_sources=2):
        """
        Args:
            in_channels <int>: Input channels
            out_channels <int>: Output channels
            kernel_size <int>: Kernel size of 1D convolution
            stride <int>: Stride of 1D convolution
            padding <int>: Padding of 1D convolution
            dilation <int>: Dilation of 1D convolution
            groups <int>: Group of 1D convolution
            bias <bool>: Applies bias to 1D convolution
            embed_dim <int>: Embedding dimension
            bottleneck_channels <int>: Bottleneck channels
            n_sources <int>: Number of sources
        """
        super().__init__()

        self.in_channels, self.out_channels = in_channels, out_channels

        self.kernel_size, self.stride = kernel_size, stride
        self.padding, self.dilation = padding, dilation
        self.groups = groups
        self.bias = bias
        self.n_sources = n_sources
        
        self.bottleneck = nn.Linear(embed_dim, bottleneck_channels)
        self.linear = nn.Linear(bottleneck_channels, out_channels*in_channels//groups*kernel_size)
        self.linear_bias = nn.Linear(bottleneck_channels, out_channels)

    def forward(self, input, embedding):
        """
        Arguments:
            input <torch.Tensor>: (batch_size, n_sources, C_in, T_in)
            embedding <torch.Tensor>: (n_sources, embed_dim)
        Returns:
            output <torch.Tensor>: (batch_size, n_sources, C_out, T_out)
        """
        C_in, C_out = self.in_channels, self.out_channels
        kernel_size, stride = self.kernel_size, self.stride
        padding, dilation = self.padding, self.dilation
        groups = self.groups
        n_sources = self.n_sources

        batch_size, _, _, T_in = input.size()

        x_embedding = self.bottleneck(embedding)  # (n_sources, bottleneck_channels)
        kernel = self.linear(x_embedding)
        kernel = kernel.view(n_sources * C_out, C_in//groups, kernel_size)

        x = input.view(batch_size, n_sources * C_in, T_in)  # shape: (batch_size, n_sources * C_in, T_in)
        x = F.conv1d(x, kernel, bias=None, stride=stride, padding=padding, dilation=dilation, groups=n_sources*groups)  # shape: (B, n_sources * C_out, T_out)
        x = x.view(batch_size, n_sources, C_out, -1)

        if self.bias:
            bias = self.linear_bias(x_embedding)
            bias = bias.view(1, n_sources, C_out, 1)
            output = x + bias  # (batch_size, n_sources, C_out, T_out)
        else:
            output = x

        return output

class Conv1dStatic(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, n_sources=2):
        """
        Args:
            in_channels <int>: Input channels
            out_channels <int>: Output channels
            kernel_size <int>: Kernel size of 1D convolution
            stride <int>: Stride of 1D convolution
            padding <int>: Padding of 1D convolution
            dilation <int>: Dilation of 1D convolution
            groups <int>: Group of 1D convolution
            bias <bool>: Applies bias to 1D convolution
            bottleneck_channels <int>: Bottleneck channels
            n_sources <int>: Number of sources
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_sources = n_sources
        
        self.conv1d = nn.Conv1d(n_sources*in_channels, n_sources*out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=n_sources*groups, bias=bias)

    def forward(self, input):
        """
        Args:
            input (batch_size, n_sources, C_in, T_in)
        Returns:
            output (batch_size, n_sources, C_out, T_out)
        """
        C_in, C_out = self.in_channels, self.out_channels
        n_sources = self.n_sources

        batch_size, _, _, T_in = input.size()

        x = input.view(batch_size, n_sources*C_in, T_in)  # (batch_size, n_sources*C_in, T_in)
        x = self.conv1d(x)  # (batch_size, n_sources*C_out, T_out)
        output = x.view(batch_size, n_sources, C_out, -1)  # (batch_size, n_sources, C_out, T_out)

        return output

def _test_conv1d():
    batch_size, n_sources = 2, 4
    C_in, T_in = 3, 10
    C_out = 5
    embed_dim = 8

    input, embedding = torch.randn(batch_size, n_sources, C_in, T_in), torch.randn(n_sources, embed_dim)
    conv1d = Conv1dGenerated(C_in, C_out, kernel_size=3, embed_dim=embed_dim, bottleneck_channels=6, n_sources=n_sources)
    output = conv1d(input, embedding)

    print(conv1d)
    print(input.size(), embedding.size(), output.size())

    input = torch.randn(batch_size, n_sources, C_in, T_in)
    conv1d = Conv1dStatic(C_in, C_out, kernel_size=3, n_sources=n_sources)
    output = conv1d(input)

    print(conv1d)
    print(input.size(), output.size())

if __name__ == '__main__':
    torch.manual_seed(111)

    _test_conv1d()