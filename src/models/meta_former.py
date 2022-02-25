import torch.nn as nn
from torch.nn.modules.utils import _pair

"""
Reference:
    "MetaFormer is Actually What You Need for Vision"
    See https://arxiv.org/abs/2111.11418
"""

class PatchEmbedding2d(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size, channel_last=True):
        super().__init__()

        self.in_channels, self.out_channels = in_channels, out_channels
        self.patch_size = _pair(patch_size)

        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=patch_size, stride=patch_size)

        self.channel_last = channel_last

    def forward(self, input):
        """
        Args:
            input: (batch_size, in_channels, height, width)
        Returns:
            output: (batch_size, out_channels, height // patch_height, width // patch_width)
        """
        out_channels = self.out_channels
        patch_height, patch_width = self.patch_size
        batch_size, _, height, width = input.size()

        assert height % patch_height == 0 and width % patch_width == 0

        num_patches = (height // patch_height) * (width // patch_width)

        x = self.conv2d(input)
        x = x.view(batch_size, out_channels, num_patches)

        if self.channel_last:
            x = x.permute(0, 2, 1)
            output = x.contiguous()
        else:
            output = x

        return output
