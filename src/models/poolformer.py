import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from models.metaformer import OverlappedPatchEmbedding2d, ChannelMixerBlock2d, choose_pool2d

EPS = 1e-12

class PoolFormer(nn.Module):
    """
    Pool Former
    Reference:
        "MetaFormer is Actually What You Need for Vision"
        See https://arxiv.org/abs/2111.11418
    """
    pretrained_model_ids = {
        "imagenet": {
            "S/36": {},
            "S/24": {},
            "S/12": {},
            "M/48": {},
            "M/36": {}
        }
    }
    def __init__(
        self,
        in_channels,
        embed_dim, hidden_channels,
        patch_size=7, stride=4,
        down_patch_size=3, down_stride=2,
        pool_size=3,
        num_layers=3,
        dropout=0,
        activation="gelu",
        skip_token=False, skip_channel=True,
        layer_scale=None,
        pooling="avg",
        bias_head=True,
        num_classes=1000,
        eps=EPS
    ):
        super().__init__()

        self.in_channels = in_channels
        patch_size = _pair(patch_size)

        if type(embed_dim) is list:
            embed_dim_in = embed_dim[0]
            embed_dim_out = embed_dim[-1]
        else:
            embed_dim_out = embed_dim_in = embed_dim

        self.patch_embedding2d = OverlappedPatchEmbedding2d(
            in_channels, embed_dim_in,
            patch_size=patch_size, stride=stride,
            channel_last=False, to_1d=False
        )
        self.backbone = PoolFormerBackbone(
            embed_dim,
            hidden_channels,
            patch_size=down_patch_size, stride=down_stride,
            pool_size=pool_size,
            num_layers=num_layers,
            dropout=dropout, activation=activation,
            norm_first=True, skip_token=skip_token, skip_channel=skip_channel,
            layer_scale=layer_scale,
            eps=eps
        )
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

    @classmethod
    def build_from_pretrained(cls, load_state_dict=True, **kwargs):
        if load_state_dict:
            raise ValueError("Not support load_state_dict=True.")

        task = kwargs.get('task')

        if not task in cls.pretrained_model_ids:
            raise KeyError("Invalid task ({}) is specified.".format(task))

        if task == "imagenet":
            specification = kwargs.get("specification") or "M/36"

            in_channels = 3

            patch_size, stride = 7, 4
            down_patch_size, down_stride = 3, 2
            pool_size = 3

            dropout = 0
            layer_scale = 1e-5
            activation = "gelu"
            pooling = "avg"
            bias_head = True
            num_classes = 1000

            total_layers = int(specification[2:])

            if specification[0] == "S":
                embed_dim = [64, 128, 320, 512]
                hidden_channels = [256, 512, 1280, 2048]

                if total_layers == 12:
                    num_layers = [2, 2, 6, 2]
                    layer_scale = 1e-5
                elif total_layers == 24:
                    num_layers = [4, 4, 12, 4]
                    layer_scale = 1e-5
                elif total_layers == 36:
                    num_layers = [6, 6, 18, 6]
                    layer_scale = 1e-6
                else:
                    raise ValueError("Not support S/{}.".format(total_layers))
            elif specification[0] == "M":
                embed_dim = [96, 192, 384, 768]
                hidden_channels = [384, 768, 1536, 3072]
                layer_scale = 1e-6

                if total_layers == 36:
                    num_layers = [6, 6, 18, 6]
                elif total_layers == 48:
                    num_layers = [8, 8, 24, 8]
                else:
                    raise ValueError("Not support M/{}.".format(total_layers))
            else:
                raise ValueError("Not support {}/*.".format(specification[0]))
        else:
            raise ValueError("Not support task={}.".format(task))

        model = cls(
            in_channels, embed_dim, hidden_channels,
            patch_size=patch_size, stride=stride,
            down_patch_size=down_patch_size, down_stride=down_stride,
            pool_size=pool_size,
            num_layers=num_layers,
            dropout=dropout, activation=activation,
            layer_scale=layer_scale,
            pooling=pooling,
            bias_head=bias_head,
            num_classes=num_classes
        )
    
        return model

class PoolFormerBackbone(nn.Module):
    def __init__(
        self,
        embed_dim,
        hidden_channels,
        patch_size=3, stride=2,
        pool_size=3,
        num_layers=6,
        dropout=0, activation="gelu",
        norm_first=True, skip_token=False, skip_channel=True,
        layer_scale=None,
        eps=EPS
    ):
        super().__init__()

        if type(num_layers) is list:
            num_blocks = num_layers
        else:
            num_blocks = [1] * num_layers

        if type(embed_dim) is list:
            _embed_dim = embed_dim
        else:
            _embed_dim = [embed_dim] * len(num_blocks)

        if type(hidden_channels) is list:
            _hidden_channels = hidden_channels
        else:
            _hidden_channels = [hidden_channels] * len(num_blocks)

        if type(patch_size) is list:
            _patch_size = patch_size
        else:
            _patch_size = [patch_size] * (len(num_blocks) - 1)

        if type(stride) is list:
            _stride = stride
        else:
            _stride = [stride] * (len(num_blocks) - 1)

        net = []

        for block_idx, _num_layers in enumerate(num_blocks):
            for layer_idx in range(_num_layers):
                if (block_idx == 0 and layer_idx == 0) or layer_idx != 0:
                    module = PoolFormerBlock(
                        _embed_dim[block_idx], _hidden_channels[block_idx],
                        patch_size=None, stride=None,
                        pool_size=pool_size,
                        dropout=dropout,
                        activation=activation,
                        norm_first=norm_first, skip_token=skip_token, skip_channel=skip_channel,
                        layer_scale=layer_scale,
                        eps=eps
                    )
                else:
                    module = PoolFormerBlock(
                        _embed_dim[block_idx-1], _hidden_channels[block_idx],
                        out_channels=_embed_dim[block_idx],
                        patch_size=_patch_size[block_idx-1], stride=_stride[block_idx-1],
                        pool_size=pool_size,
                        dropout=dropout,
                        activation=activation,
                        norm_first=norm_first, skip_token=skip_token, skip_channel=skip_channel,
                        layer_scale=layer_scale,
                        eps=eps
                    )
                net.append(module)

        self.net = nn.Sequential(*net)

    def forward(self, input):
        """
        Args:
            input: (batch_size, num_patches, embed_dim)
        Returns:
            output: (batch_size, num_patches, embed_dim)
        """
        output = self.net(input)

        return output

class PoolFormerBlock(nn.Module):
    def __init__(
        self,
        in_channels, hidden_channels,
        patch_size=None, stride=None,
        pool_size=3,
        dropout=0, activation="gelu",
        out_channels=None,
        norm_first=True, skip_token=False, skip_channel=True,
        layer_scale=None,
        eps=EPS
    ):
        super().__init__()

        self.patch_size, self.stride = patch_size, stride

        if patch_size is None:
            self.patch_embedding2d = None
            if out_channels is not None and out_channels != in_channels:
                raise ValueError("out_channels should be in_channels when patch_size is None.")
            out_channels = in_channels
        else:
            self.patch_size = _pair(self.patch_size)
            self.stride = _pair(self.stride)

            if out_channels is None:
                out_channels = in_channels

            self.patch_embedding2d = OverlappedPatchEmbedding2d(
                in_channels, out_channels,
                patch_size=patch_size, stride=stride,
                channel_last=False, to_1d=False
            )

        self.token_mixer = TokenMixerBlock2d(
            out_channels, pool_size=pool_size,
            norm_first=norm_first, channel_last=False,
            eps=eps
        )

        self.channel_mixer = ChannelMixerBlock2d(
            out_channels, hidden_channels,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first, channel_last=False,
            eps=eps
        )

        if layer_scale is None:
            self.layer_scale = False
        else:
            self.token_scale = nn.Parameter(layer_scale * torch.ones(out_channels), requires_grad=True)
            self.channel_scale = nn.Parameter(layer_scale * torch.ones(out_channels), requires_grad=True)
            self.layer_scale = True

        self.skip_token = skip_token
        self.skip_channel = skip_channel

    def forward(self, input):
        """
        Args:
            input: (batch_size, embed_dim, height, width)
        Returns:
            output: (batch_size, embed_dim, height', width')
        """
        if self.patch_embedding2d is None:
            x = input
        else:
            Kh, Kw = self.patch_size
            Sh, Sw = self.stride
            Ph, Pw = Kh - Sh, Kw - Sw
            Ph_top, Pw_left = Ph // 2, Pw // 2
            Ph_bottom, Pw_right = Ph - Ph_top, Pw - Pw_left

            x = F.pad(input, (Pw_left, Pw_right, Ph_top, Ph_bottom))
            x = self.patch_embedding2d(x)

        if self.layer_scale:
            x_rescale = self.token_scale.view(-1, 1, 1) * self.token_mixer(x)
        else:
            x_rescale = self.token_mixer(x)

        if self.skip_token:
            x = x + x_rescale # (batch_size, embed_dim, height', width')
        else:
            x = x_rescale # (batch_size, embed_dim, height', width')

        if self.layer_scale:
            x_rescale = self.channel_scale.view(-1, 1, 1) * self.channel_mixer(x)
        else:
            x_rescale = self.channel_mixer(x)

        if self.skip_channel:
            output = x + x_rescale # (batch_size, embed_dim, height', width')
        else:
            output = x_rescale # (batch_size, embed_dim, height', width')

        return output

class TokenMixerBlock2d(nn.Module):
    def __init__(self, num_features, pool_size, norm_first=True, channel_last=False, eps=EPS):
        super().__init__()

        assert norm_first, "norm_first should be True."
        assert not channel_last, "channel_first should be False."

        pool_size = _pair(pool_size)

        self.layer_norm = nn.GroupNorm(1, num_features, eps=eps)
        self.mixer = nn.AvgPool2d(pool_size, stride=1)
        self.pool_size = pool_size

    def forward(self, input):
        """
        Args:
            input: (batch_size, num_features, height, width)
        Returns:
            output: (batch_size, num_features, height, width)
        """
        Kh, Kw = self.pool_size
        Ph, Pw = Kh - 1, Kw - 1
        Ph_top, Pw_left = Ph // 2, Pw // 2
        Ph_bottom, Pw_right = Ph - Ph_top, Pw - Pw_left

        x = self.layer_norm(input)
        x = F.pad(x, (Pw_left, Pw_right, Ph_top, Ph_bottom))
        output = self.mixer(x)

        return output

def _test_poolformer():
    in_channels = 3
    embed_dim = [8, 5, 8]
    hidden_channels = [12, 16, 18]
    image_size = 224
    patch_size, stride = 7, 4
    down_patch_size, down_stride = 3, 2
    pool_size = 3
    num_layers = [2, 3, 2]
    dropout = 0
    activation = "gelu"
    pooling = "avg"
    bias_head = True
    num_classes = 1000

    model = PoolFormer(
        in_channels, embed_dim, hidden_channels,
        patch_size=patch_size, stride=stride,
        down_patch_size=down_patch_size, down_stride=down_stride,
        pool_size=pool_size,
        num_layers=num_layers,
        dropout=dropout,
        activation=activation,
        pooling=pooling,
        bias_head=bias_head,
        num_classes=num_classes
    )

    print(model)
    input = torch.randn(4, in_channels, image_size, image_size)
    output = model(input)

    print(model)
    print(input.size(), output.size())

if __name__ == '__main__':
    import torch

    torch.manual_seed(111)

    print("="*10, "PoolFormer", "="*10)
    _test_poolformer()