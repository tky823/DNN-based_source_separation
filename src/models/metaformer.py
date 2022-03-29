import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from utils.model import choose_nonlinear
from modules.pool import GlobalAvgPool1d, GlobalAvgPool2d

EPS = 1e-12

"""
Reference:
    "MetaFormer is Actually What You Need for Vision"
    See https://arxiv.org/abs/2111.11418
"""
class MetaFormer(nn.Module):
    """
    Meta Former
    Reference:
        "MetaFormer is Actually What You Need for Vision"
        See https://arxiv.org/abs/2111.11418
    """
    def __init__(
        self,
        backbone,
        in_channels, embed_dim_in, embed_dim_out,
        patch_size=7, stride=4,
        pooling="avg",
        bias_head=True,
        num_classes=1000,
        eps=EPS
    ):
        super().__init__()

        self.in_channels = in_channels
        patch_size = _pair(patch_size)

        self.patch_embedding2d = OverlappedPatchEmbedding2d(
            in_channels, embed_dim_in,
            patch_size=patch_size, stride=stride,
            channel_last=False, to_1d=False
        )
        self.backbone = backbone
        self.norm2d = nn.GroupNorm(1, embed_dim_out, eps=eps)
        self.pool2d = choose_pool2d(pooling, channel_last=False)
        self.fc_head = nn.Linear(embed_dim_out, num_classes, bias=bias_head)

    def forward(self, input):
        """
        Args:
            input: (batch_size, in_channels, height, width)
        Returns:
            output: (batch_size, num_classes)
        """
        x = self.patch_embedding2d(input) # (batch_size, embed_dim_in, H', W')
        x = self.backbone(x) # (batch_size, embed_dim_out, H'', W'')
        x = self.norm2d(x) # (batch_size, embed_dim_out, H'', W'')
        x = self.pool2d(x) # (batch_size, embed_dim_out)
        output = self.fc_head(x) # (batch_size, num_classes)

        return output

class MetaFormerBackbone(nn.Module):
    def __init__(self, layer: nn.Module, num_layers: int, norm=None):
        super().__init__()

        self.net = _get_clones(layer, num_layers)
        self.num_layers = num_layers

        self.norm = norm

    def forward(self, input):
        x = input

        for idx in range(self.num_layers):
            x = self.net[idx](x)

        if self.norm is None:
            output = x
        else:
            output = self.norm(x)

        return output

class ChannelMixerBlock1d(nn.Module):
    def __init__(self, num_features, hidden_channels, dropout=0, activation="gelu", norm_first=True, channel_last=True, eps=EPS):
        super().__init__()

        assert norm_first, "norm_first should be True."

        self.layer_norm = nn.LayerNorm(num_features, eps=eps)
        self.mixer = MLPBlock1d(num_features, hidden_channels, dropout=dropout, activation=activation, channel_last=channel_last)

    def forward(self, input):
        """
        Args:
            input: (batch_size, token_size, num_features)
        Returns:
            output: (batch_size, token_size, num_features)
        """
        x = self.layer_norm(input)
        output = self.mixer(x)

        return output

class ChannelMixerBlock2d(nn.Module):
    def __init__(self, num_features, hidden_channels, dropout=0, activation="gelu", norm_first=True, channel_last=False, eps=EPS):
        super().__init__()

        assert norm_first, "norm_first should be True."

        self.layer_norm = nn.GroupNorm(1, num_features, eps=eps)
        self.mixer = MLPBlock2d(num_features, hidden_channels, dropout=dropout, activation=activation, channel_last=channel_last)

    def forward(self, input):
        """
        Args:
            input: (batch_size, num_features, height, width)
        Returns:
            output: (batch_size, num_features, height, width)
        """
        x = self.layer_norm(input)
        output = self.mixer(x)

        return output

class MLPBlock1d(nn.Module):
    def __init__(self, num_features, hidden_channels, dropout=0, activation="gelu", channel_last=True):
        super().__init__()

        assert channel_last, "channel_last should be True."

        self.linear1 = nn.Linear(num_features, hidden_channels)
        self.activation = choose_nonlinear(activation)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_channels, num_features)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, input):
        """
        Args:
            input: (batch_size, length, num_features)
        Returns:
            output: (batch_size, length, num_features)
        """
        x = self.linear1(input)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        output = self.dropout2(x)

        return output

class MLPBlock2d(nn.Module):
    def __init__(self, num_features, hidden_channels, dropout=0, activation="gelu", channel_last=False):
        super().__init__()

        assert not channel_last, "channel_last should be False."

        self.linear1 = nn.Conv2d(num_features, hidden_channels, kernel_size=1, stride=1)
        self.activation = choose_nonlinear(activation)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Conv2d(hidden_channels, num_features, kernel_size=1, stride=1)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, input):
        """
        Args:
            input: (batch_size, num_features, height, width)
        Returns:
            output: (batch_size, num_features, height, width)
        """
        x = self.linear1(input)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        output = self.dropout2(x)

        return output

class PatchEmbedding2d(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size, channel_last=True, to_1d=True):
        super().__init__()

        self.in_channels, self.embed_dim = in_channels, embed_dim
        self.patch_size = _pair(patch_size)

        self.conv2d = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

        self.channel_last = channel_last
        self.to_1d = to_1d

    def forward(self, input):
        """
        Args:
            input: (batch_size, in_channels, height, width)
        Returns:
            output:
                (batch_size, num_patches, embed_dim) if channel_last = True and to_1d = True
                (batch_size, height', width', embed_dim) if channel_last = True and to_1d = False
                (batch_size, embed_dim, num_patches) if channel_last = False and to_1d = True
                (batch_size, embed_dim, height', width') if channel_last = False and to_1d = False
                where height' = height // patch_height and width' = width // patch_width
                where num_patches = height' * width'
        """
        patch_height, patch_width = self.patch_size
        _, _, height, width = input.size()

        assert height % patch_height == 0 and width % patch_width == 0

        x = self.conv2d(input)

        if self.to_1d:
            x = torch.flatten(x, start_dim=2) # (batch_size, embed_dim, num_patches), where num_patches = (height // patch_height) * (width // patch_width)
            if self.channel_last:
                x = x.permute(0, 2, 1)
                output = x.contiguous()
            else:
                output = x
        else:
            if self.channel_last:
                x = x.permute(0, 2, 3, 1)
                output = x.contiguous()
            else:
                output = x

        return output

class OverlappedPatchEmbedding2d(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size, stride=None, channel_last=True, to_1d=True):
        super().__init__()

        if stride is None:
            raise ValueError("Use PatchEmbedding2d or specify stride.")

        self.in_channels, self.embed_dim = in_channels, embed_dim
        self.patch_size = _pair(patch_size)
        self.stride = _pair(stride)

        self.conv2d = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=stride)

        self.channel_last = channel_last
        self.to_1d = to_1d

    def forward(self, input):
        """
        Args:
            input: (batch_size, in_channels, height, width)
        Returns:
            output:
                (batch_size, num_patches, embed_dim) if channel_last = True and to_1d = True
                (batch_size, height', width', embed_dim) if channel_last = True and to_1d = False
                (batch_size, embed_dim, num_patches) if channel_last = False and to_1d = True
                (batch_size, embed_dim, height', width') if channel_last = False and to_1d = False
                where height' = height // stride_height and width' = width // stride_width
                where num_patches = height' * width'
        """
        Kh, Kw = self.patch_size
        Sh, Sw = self.stride
        Ph, Pw = Kh - Sh, Kw - Sw
        Ph_top, Pw_left = Ph // 2, Pw // 2
        Ph_bottom, Pw_right = Ph - Ph_top, Pw - Pw_left

        x = F.pad(input, (Pw_left, Pw_right, Ph_top, Ph_bottom))
        x = self.conv2d(x)

        if self.to_1d:
            x = torch.flatten(x, start_dim=2) # (batch_size, embed_dim, num_patches), where num_patches = (height // stride_height) * (width // stride_width)
            if self.channel_last:
                x = x.permute(0, 2, 1)
                output = x.contiguous()
            else:
                output = x
        else:
            if self.channel_last:
                x = x.permute(0, 2, 3, 1)
                output = x.contiguous()
            else:
                output = x

        return output

class Pool1d(nn.Module):
    def __init__(self, pooling, channel_last=True):
        super().__init__()

        if pooling == "avg":
            self.module = GlobalAvgPool1d(keepdim=False)
        else:
            raise ValueError("Not suuport {}-pooling.".format(pooling))

        self.channel_last = channel_last

    def forward(self, input):
        if self.channel_last:
            x = input.permute(0, 2, 1)
        else:
            x = input

        output = self.module(x)

        return output

class Pool2d(nn.Module):
    def __init__(self, pooling, channel_last=False):
        super().__init__()

        if pooling == "avg":
            self.module = GlobalAvgPool2d(keepdim=False)
        else:
            raise ValueError("Not suuport {}-pooling.".format(pooling))

        self.channel_last = channel_last

    def forward(self, input):
        if self.channel_last:
            x = input.permute(0, 3, 1, 2)
        else:
            x = input

        output = self.module(x)

        return output

def choose_pool1d(args, channel_last=True):
    if type(args) is str:
        module = Pool1d(pooling=args, channel_last=channel_last)
    elif isinstance(args, nn.Module):
        module = args
    else:
        raise ValueError("Not support {}-pooling.".format(args))

    return module

def choose_pool2d(args, channel_last=False):
    if type(args) is str:
        module = Pool2d(pooling=args, channel_last=channel_last)
    elif isinstance(args, nn.Module):
        module = args
    else:
        raise ValueError("Not support {}-pooling.".format(args))

    return module

def _get_clones(module, N):
    return nn.Sequential(*[copy.deepcopy(module) for _ in range(N)])

def _test_patch_embedding():
    in_channels, embed_dim = 3, 64
    image_size, patch_size = 224, 16

    input = torch.randn(4, in_channels, image_size, image_size)

    print("-"*10, "PatchEmbedding2d (channel_last=True, to_1d=True)", "-"*10)
    model = PatchEmbedding2d(
        in_channels, embed_dim,
        patch_size=patch_size,
        channel_last=True, to_1d=True
    )
    output = model(input)

    print(model)
    print(input.size(), output.size())
    print()

    print("-"*10, "PatchEmbedding2d (channel_last=True, to_1d=False)", "-"*10)
    model = PatchEmbedding2d(
        in_channels, embed_dim,
        patch_size=patch_size,
        channel_last=True, to_1d=False
    )
    output = model(input)

    print(model)
    print(input.size(), output.size())
    print()

    print("-"*10, "PatchEmbedding2d (channel_last=False, to_1d=True)", "-"*10)
    model = PatchEmbedding2d(
        in_channels, embed_dim,
        patch_size=patch_size,
        channel_last=False, to_1d=True
    )
    output = model(input)

    print(model)
    print(input.size(), output.size())
    print()

    print("-"*10, "PatchEmbedding2d (channel_last=False, to_1d=False)", "-"*10)
    model = PatchEmbedding2d(
        in_channels, embed_dim,
        patch_size=patch_size,
        channel_last=False, to_1d=False
    )
    output = model(input)

    print(model)
    print(input.size(), output.size())
    print()

def _test_overlapped_patch_embedding():
    in_channels, embed_dim = 3, 64
    image_size, patch_size, stride = 224, 3, 2

    input = torch.randn(4, in_channels, image_size, image_size)

    print("-"*10, "OverlappedPatchEmbedding2d (channel_last=True, to_1d=True)", "-"*10)
    model = OverlappedPatchEmbedding2d(
        in_channels, embed_dim,
        patch_size=patch_size, stride=stride,
        channel_last=True, to_1d=True
    )
    output = model(input)

    print(model)
    print(input.size(), output.size())
    print()

    print("-"*10, "OverlappedPatchEmbedding2d (channel_last=True, to_1d=False)", "-"*10)
    model = OverlappedPatchEmbedding2d(
        in_channels, embed_dim,
        patch_size=patch_size, stride=stride,
        channel_last=True, to_1d=False
    )
    output = model(input)

    print(model)
    print(input.size(), output.size())
    print()

    print("-"*10, "OverlappedPatchEmbedding2d (channel_last=False, to_1d=True)", "-"*10)
    model = OverlappedPatchEmbedding2d(
        in_channels, embed_dim,
        patch_size=patch_size, stride=stride,
        channel_last=False, to_1d=True
    )
    output = model(input)

    print(model)
    print(input.size(), output.size())
    print()

    print("-"*10, "OverlappedPatchEmbedding2d (channel_last=False, to_1d=False)", "-"*10)
    model = OverlappedPatchEmbedding2d(
        in_channels, embed_dim,
        patch_size=patch_size, stride=stride,
        channel_last=False, to_1d=False
    )
    output = model(input)

    print(model)
    print(input.size(), output.size())
    print()

def _test_mlp_mixer():
    from models.mlp_mixer import MLPMixerBlock2d

    in_channels = 3
    embed_dim, hidden_channels = 8, 12
    image_size, patch_size = 224, 16
    num_layers = 2

    mixer_block = MLPMixerBlock2d(
        embed_dim, token_hidden_channels=hidden_channels, embed_hidden_channels=hidden_channels,
        num_patches=(image_size//patch_size)**2,
    )
    backbone = MetaFormerBackbone(mixer_block, num_layers=num_layers, norm=None)

    model = MetaFormer(
        backbone,
        in_channels, embed_dim_in=embed_dim, embed_dim_out=embed_dim,
        patch_size=patch_size, stride=patch_size,
        num_classes=10
    )

    input = torch.randn(4, in_channels, image_size, image_size)
    output = model(input)

    print(model)
    print(input.size(), output.size())

def _test_poolformer():
    from models.poolformer import PoolFormerBackbone

    in_channels = 3
    embed_dim, hidden_channels = [5, 6, 7], [12, 16, 18]
    down_patch_size, down_stride = 3, 2
    pool_size = 3
    num_layers = [2, 3, 2]

    backbone = PoolFormerBackbone(
        embed_dim, hidden_channels,
        patch_size=down_patch_size, stride=down_stride,
        pool_size=pool_size,
        num_layers=num_layers,
        layer_scale=1e-5
    )
    model = MetaFormer(
        backbone,
        in_channels, embed_dim_in=embed_dim[0], embed_dim_out=embed_dim[-1],
        num_classes=10
    )

    input = torch.randn(4, in_channels, 256, 256)
    output = model(input)

    print(model)
    print(input.size(), output.size())

if __name__ == "__main__":
    torch.manual_seed(111)

    print("="*10, "PatchEmbedding", "="*10)
    _test_patch_embedding()

    print("="*10, "OverlappedPatchEmbedding", "="*10)
    _test_overlapped_patch_embedding()

    print("="*10, "MLP-Mixer", "="*10)
    _test_mlp_mixer()

    print("="*10, "PoolFormer", "="*10)
    _test_poolformer()