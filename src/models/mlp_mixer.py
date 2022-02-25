import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from utils.model import choose_nonlinear
from models.vit import PatchEmbedding2d

EPS = 1e-12

class MLPMixer(nn.Module):
    pretrained_model_ids = {
        "imagenet": {}
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

        image_size = _pair(image_size)
        patch_size = _pair(patch_size)

        H, W = image_size
        pH, pW = patch_size

        num_patches = (H // pH) * (W // pW)

        self.patch_embedding2d = PatchEmbedding2d(in_channels, embed_dim, patch_size=patch_size, channel_last=True)
        self.backbone = MLPMixerBackbone(
            embed_dim,
            token_hidden_channels, embed_hidden_channels,
            num_patches,
            num_layers=num_layers,
            dropout=dropout, activation=activation,
            norm_first=True
        )

        # Order of modules ?
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

            eps = EPS
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
            num_classes=num_classes,
            eps=eps
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
            module = MixerBlock(
                embed_dim, token_hidden_channels, embed_hidden_channels, num_patches,
                dropout=dropout,
                activation=activation,
                norm_first=norm_first
            )
            net.append(module)

        self.net = nn.Sequential(*net)

    def forward(self, input):
        output = self.net(input)

        return output

class MixerBlock(nn.Module):
    def __init__(self, embed_dim, token_hidden_channels, embed_hidden_channels, num_patches, dropout=0, activation="gelu", norm_first=True):
        super().__init__()

        self.token_mixer = TokenMixer(
            num_patches, token_hidden_channels,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first
        )
        self.channel_mixer = ChannelMixer(
            embed_dim, embed_hidden_channels,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first
        )

    def forward(self, input):
        x = input + self.token_mixer(input)
        output = x + self.channel_mixer(x)

        return output

class TokenMixer(nn.Module):
    def __init__(self, num_features, hidden_channels, dropout=0, activation="gelu", norm_first=True):
        super().__init__()

        assert norm_first, "norm_first should be True."

        self.linear1 = nn.Linear(num_features, hidden_channels)
        self.activation = choose_nonlinear(activation)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_channels, num_features)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(num_features)

        self.norm_first = norm_first

    def forward(self, input):
        """
        Args:
            input: (batch_size, num_patches, num_features)
        Returns:
            output: (batch_size, num_patches, num_features)
        """
        x = self.layer_norm(input)
        x = x.permute(0, 2, 1)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        output = x.permute(0, 2, 1)

        return output

class ChannelMixer(nn.Module):
    def __init__(self, num_features, hidden_channels, dropout=0, activation="gelu", norm_first=True):
        super().__init__()

        assert norm_first, "norm_first should be True."

        self.linear1 = nn.Linear(num_features, hidden_channels)
        self.activation = choose_nonlinear(activation)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_channels, num_features)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(num_features)

        self.norm_first = norm_first

    def forward(self, input):
        """
        Args:
            input: (batch_size, token_size, num_features)
        Returns:
            output: (batch_size, token_size, num_features)
        """
        x = self.layer_norm(input)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        output = self.dropout2(x)

        return output

class MLPMixerPool1d(nn.Module):
    def __init__(self, pooling="avg"):
        super().__init__()

        self.pooling = pooling

        if not self.pooling in ["avg", "max"]:
            raise ValueError("Not support pooling={}".format(self.pooling))

    def forward(self, input):
        if self.pooling == "avg":
            output = F.adaptive_avg_pool1d(input, 1, return_indices=False)
        else:
            output = F.adaptive_max_pool1d(input, 1, return_indices=False)

        return output

def _test_mlp_mixer():
    in_channels = 3
    image_size, patch_size = 256, 16
    num_classes = 100

    d_model = 1024
    nhead, num_layers = 16, 6
    embed_dim = nhead * 64
    d_ff = 2048

    model = ViT(
        transformer,
        in_channels=in_channels, embed_dim=embed_dim,
        image_size=image_size, patch_size=patch_size,
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