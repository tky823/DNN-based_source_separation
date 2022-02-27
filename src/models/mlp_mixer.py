import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from models.metaformer import PatchEmbedding2d, ChannelMixerBlock1d, ChannelMixerBlock2d, MLPBlock1d

EPS = 1e-12

class MLPMixer(nn.Module):
    """
    MLP-Mixer
    Reference:
        "MLP-Mixer: An all-MLP Architecture for Vision"
        See https://arxiv.org/abs/2105.01601
    """
    pretrained_model_ids = {
        "imagenet": {
            "S/32": {},
            "S/16": {},
            "B/32": {},
            "B/16": {},
            "L/32": {},
            "L/16": {},
            "H/14": {}
        }
    }
    def __init__(
        self,
        in_channels,
        embed_dim,
        token_hidden_channels, embed_hidden_channels,
        image_size,
        patch_size=16,
        num_layers=8,
        dropout=0,
        activation="gelu",
        pooling="avg",
        bias_head=True,
        num_classes=1000,
        eps=EPS
    ):
        super().__init__()

        self.in_channels = in_channels
        self.image_size = _pair(image_size)
        patch_size = _pair(patch_size)

        H, W = self.image_size
        pH, pW = patch_size

        num_patches = (H // pH) * (W // pW)

        self.patch_embedding2d = PatchEmbedding2d(
            in_channels, embed_dim,
            patch_size=patch_size, channel_last=True, to_1d=True
        )
        self.backbone = MLPMixerBackbone(
            embed_dim,
            token_hidden_channels, embed_hidden_channels,
            num_patches,
            num_layers=num_layers,
            dropout=dropout, activation=activation,
            norm_first=True
        )
        self.norm1d = nn.LayerNorm(embed_dim, eps=eps)
        self.pool1d = MLPMixerPool1d(pooling)
        self.fc_head = nn.Linear(embed_dim, num_classes, bias=bias_head)

    def forward(self, input):
        """
        Args:
            input: (batch_size, in_channels, height, width)
        Returns:
            output: (batch_size, num_classes)
        """
        C = self.in_channels
        H, W = self.image_size
        _, C_in, H_in, W_in = input.size()

        assert C_in == C and H_in == H and W_in == W, \
            "Input shape is expected (batch_size, {}, {}, {}), but given (batch_size, {}, {}, {})".format(C, H, W, C_in, H_in, W_in)

        x = self.patch_embedding2d(input) # (batch_size, num_patches, embed_dim)
        x = self.backbone(x) # (batch_size, num_patches, embed_dim)
        x = self.norm1d(x) # (batch_size, num_patches, embed_dim)
        x = self.pool1d(x) # (batch_size, embed_dim)
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
            specification = kwargs.get("specification") or "B/16"

            in_channels = 3
            image_size = 224

            dropout = 0
            activation = "gelu"
            pooling = "avg"
            bias_head = True
            num_classes = 1000

            if specification[0] == "S":
                embed_dim = 512
                token_hidden_channels, embed_hidden_channels = 256, 2048
                num_layers = 8
            elif specification[0] == "B":
                embed_dim = 768
                token_hidden_channels, embed_hidden_channels = 384, 3072
                num_layers = 12
            elif specification[0] == "L":
                embed_dim = 1024
                token_hidden_channels, embed_hidden_channels = 512, 4096
                num_layers = 24
            elif specification[0] == "H":
                embed_dim = 1280
                token_hidden_channels, embed_hidden_channels = 640, 5120
                num_layers = 32
            else:
                raise ValueError("Not support {}/*.".format(specification[0]))

            patch_size = int(specification[2:])
        else:
            raise ValueError("Not support task={}.".format(task))

        model = cls(
            in_channels,
            embed_dim,
            token_hidden_channels, embed_hidden_channels,
            image_size,
            patch_size=patch_size,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
            pooling=pooling,
            bias_head=bias_head,
            num_classes=num_classes
        )

        return model

class MLPMixerBackbone(nn.Module):
    def __init__(
        self,
        embed_dim,
        token_hidden_channels, embed_hidden_channels,
        num_patches,
        num_layers=6,
        dropout=0, activation="gelu",
        norm_first=True
    ):
        super().__init__()

        net = []

        for _ in range(num_layers):
            module = MLPMixerBlock1d(
                embed_dim, token_hidden_channels, embed_hidden_channels,
                num_patches=num_patches,
                dropout=dropout,
                activation=activation,
                norm_first=norm_first
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

class MLPMixerBlock1d(nn.Module):
    def __init__(
        self,
        embed_dim, token_hidden_channels, embed_hidden_channels,
        num_patches,
        dropout=0, activation="gelu",
        norm_first=True
    ):
        super().__init__()

        self.token_mixer = TokenMixerBlock1d(
            embed_dim, num_patches, token_hidden_channels,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first, channel_last=True
        )
        self.channel_mixer = ChannelMixerBlock1d(
            embed_dim, embed_hidden_channels,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first, channel_last=True
        )

    def forward(self, input):
        """
        Args:
            input: (batch_size, num_patches, embed_dim)
        Returns:
            output: (batch_size, num_patches, embed_dim)
        """
        x = input + self.token_mixer(input) # (batch_size, num_patches, embed_dim)
        output = x + self.channel_mixer(x) # (batch_size, num_patches, embed_dim)

        return output

class MLPMixerBlock2d(nn.Module):
    def __init__(
        self,
        embed_dim, token_hidden_channels, embed_hidden_channels,
        num_patches,
        dropout=0, activation="gelu",
        norm_first=True
    ):
        super().__init__()

        self.token_mixer = TokenMixerBlock2d(
            embed_dim, num_patches, token_hidden_channels,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first, channel_last=False
        )
        self.channel_mixer = ChannelMixerBlock2d(
            embed_dim, embed_hidden_channels,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first, channel_last=False
        )

    def forward(self, input):
        """
        Args:
            input: (batch_size, num_patches, embed_dim)
        Returns:
            output: (batch_size, num_patches, embed_dim)
        """
        x = input + self.token_mixer(input) # (batch_size, num_patches, embed_dim)
        output = x + self.channel_mixer(x) # (batch_size, num_patches, embed_dim)

        return output

class TokenMixerBlock1d(nn.Module):
    def __init__(self, num_features, num_patches, hidden_channels, dropout=0, activation="gelu", norm_first=True, channel_last=True, eps=EPS):
        super().__init__()

        assert norm_first, "norm_first should be True."

        self.layer_norm = nn.LayerNorm(num_features, eps=eps)
        self.mixer = MLPBlock1d(num_patches, hidden_channels, dropout=dropout, activation=activation, channel_last=channel_last)

    def forward(self, input):
        """
        Args:
            input: (batch_size, num_patches, num_features)
        Returns:
            output: (batch_size, num_patches, num_features)
        """
        x = self.layer_norm(input)
        x = x.permute(0, 2, 1)
        x = self.mixer(x)
        output = x.permute(0, 2, 1)

        return output

class TokenMixerBlock2d(nn.Module):
    def __init__(self, num_features, num_patches, hidden_channels, dropout=0, activation="gelu", norm_first=True, channel_last=False, eps=EPS):
        super().__init__()

        assert norm_first, "norm_first should be True."

        if channel_last:
            self.layer_norm = nn.LayerNorm(num_features, eps=eps)
        else:
            self.layer_norm = nn.GroupNorm(1, num_features, eps=eps)

        self.mixer = MLPBlock1d(num_patches, hidden_channels, dropout=dropout, activation=activation, channel_last=True)

        self.channel_last = channel_last

    def forward(self, input):
        """
        Args:
            input:
                (batch_size, height, width, num_features)
                (batch_size, num_features, height, width)
                num_patches == height * width
        Returns:
            output:
                (batch_size, height, width, num_features)
                (batch_size, num_features, height, width)
        """
        if self.channel_last:
            batch_size, height, width, num_features = input.size()
            x = self.layer_norm(input)
            x = x.view(batch_size, height * width, num_features)
            x = x.permute(0, 2, 1) # (batch_size, num_features, height * width)
        else:
            batch_size, num_features, height, width = input.size()
            x = self.layer_norm(input)
            x = x.view(batch_size, num_features, height * width)

        x = self.mixer(x)

        if self.channel_last:
            x = x.permute(0, 2, 1) # (batch_size, height * width, num_features)
            output = x.view(batch_size, height, width, num_features)
        else:
            output = x.view(batch_size, num_features, height, width)

        return output

class MLPMixerPool1d(nn.Module):
    def __init__(self, pooling="avg"):
        super().__init__()

        self.pooling = pooling

        if not self.pooling in ["avg", "max"]:
            raise ValueError("Not support pooling={}".format(self.pooling))

    def forward(self, input):
        """
        Args:
            input: (batch_size, length, embed_dim)
        Returns:
            output: (batch_size, embed_dim)
        """
        x = input.permute(0, 2, 1) # (batch_size, length, num_patches)

        if self.pooling == "avg":
            x = F.adaptive_avg_pool1d(x, 1)
        else:
            x = F.adaptive_max_pool1d(x, 1, return_indices=False)

        output = x.squeeze(dim=-1)

        return output

def _test_mlp_mixer():
    in_channels = 3
    embed_dim = 512
    token_hidden_channels, embed_hidden_channels = 256, 2048
    image_size, patch_size = 224, 16
    num_layers = 8
    dropout = 0
    activation = "gelu"
    pooling = "avg"
    bias_head = True
    num_classes = 1000

    model = MLPMixer(
        in_channels,
        embed_dim, token_hidden_channels, embed_hidden_channels,
        image_size,
        patch_size=patch_size,
        num_layers=num_layers,
        dropout=dropout,
        activation=activation,
        pooling=pooling,
        bias_head=bias_head,
        num_classes=num_classes
    )

    input = torch.randn(4, in_channels, image_size, image_size)
    output = model(input)

    print(model)
    print(input.size(), output.size())

if __name__ == '__main__':
    torch.manual_seed(111)

    print("="*10, "MLP-Mixer", "="*10)
    _test_mlp_mixer()