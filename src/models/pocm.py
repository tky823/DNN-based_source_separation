import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Reference: "LaSAFT: Latent Source Attentive Frequency Transformation For Conditioned Source Separation"
"""

class PoCM2d(nn.Module):
    """
    Point-wise Convolutional Modulation
    """
    def __init__(self):
        super().__init__()

    def forward(self, input, gamma, beta):
        """
        Args:
            input (batch_size, in_channels, n_bins, n_frames)
            gamma (batch_size, out_channels, in_channels)
            beta (batch_size, out_channels)
        Returns:
            output (batch_size, out_channels, n_bins, n_frames)
        """
        batch_size, in_channels, n_bins, n_frames = input.size()
        out_channels = gamma.size(1)

        input = input.view(1, batch_size * in_channels, n_bins, n_frames)

        gamma = gamma.view(batch_size * out_channels, in_channels, 1, 1)
        beta = beta.view(batch_size * out_channels)

        output = F.conv2d(input, gamma, bias=beta, stride=(1, 1), groups=batch_size)
        output = output.view(batch_size, out_channels, n_bins, n_frames)

        return output

class GPoCM2d(nn.Module):
    """
    Gated Point-wise Convolutional Modulation
    """
    def __init__(self):
        super().__init__()

        self.pocm = PoCM2d()

    def forward(self, input, gamma, beta):
        assert gamma.size(-2) == gamma.size(-1), "gamma is expected (batch_size, C, C), but given {}.".format(gamma.size())

        x = self.pocm(input, gamma, beta)
        output = torch.sigmoid(x) * input

        return output

def _test_pocm():
    torch.manual_seed(111)

    batch_size, in_channels = 4, 3
    out_channels = in_channels

    print("-"*10, "PoCM2d", "-"*10)

    H, W = 5, 6
    input = torch.randn((batch_size, in_channels, H, W), dtype=torch.float)
    gamma, beta = torch.randn((batch_size, out_channels, in_channels), dtype=torch.float), torch.randn((batch_size, out_channels), dtype=torch.float)
    model = PoCM2d()
    output = model(input, gamma, beta)

    print(model)
    print(input.size(), output.size())

def _test_gpocm():
    torch.manual_seed(111)

    batch_size, in_channels = 4, 3
    out_channels = in_channels

    print("-"*10, "GPoCM2d", "-"*10)

    H, W = 5, 6
    input = torch.randn((batch_size, in_channels, H, W), dtype=torch.float)
    gamma, beta = torch.randn((batch_size, out_channels, in_channels), dtype=torch.float), torch.randn((batch_size, out_channels), dtype=torch.float)
    model = GPoCM2d()
    output = model(input, gamma, beta)

    print(model)
    print(input.size(), output.size())
    print()


if __name__ == '__main__':
    _test_pocm()
    print()

    _test_gpocm()