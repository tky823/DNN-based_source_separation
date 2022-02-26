import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair

from models.metaformer import PatchEmbedding2d

EPS = 1e-12

class ViT(nn.Module):
    """
    Vision Transformer
    Reference:
        "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
        See https://arxiv.org/abs/2010.11929
    """
    pretrained_model_ids = {
        "imagenet": {
            "B/32": {},
            "B/16": {},
            "L/32": {},
            "L/16": {},
            "H/14": {}
        }
    }
    def __init__(
        self,
        transformer,
        in_channels, embed_dim,
        image_size,
        patch_size=16,
        dropout=0,
        pooling="cls",
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

        self.patch_embedding2d = PatchEmbedding2d(
            in_channels, embed_dim,
            patch_size=patch_size, channel_last=True, to_1d=True
        )
        self.dropout1d = nn.Dropout(p=dropout)
        self.transformer = transformer
        self.norm1d = nn.LayerNorm(embed_dim, eps=eps)
        self.pool1d = ViTPool(pooling, dim=1)
        self.fc_head = nn.Linear(embed_dim, num_classes, bias=bias_head)

        self.positional_embedding = nn.Parameter(torch.empty(num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.empty(1, 1, embed_dim))

        self._reset_parameters()

    def _reset_parameters(self):
        self.positional_embedding.data.normal_()
        self.cls_token.data.normal_()

    def forward(self, input):
        """
        Args:
            input: (batch_size, in_channels, height, width)
        Returns:
            output: (batch_size, num_classes)
        """
        batch_size = input.size(0)

        cls_tokens = self.cls_token.repeat(batch_size, 1, 1) # (batch_size, num_patches, embed_dim)

        x = self.patch_embedding2d(input) # (batch_size, num_patches, embed_dim)
        x = torch.cat([cls_tokens, x], dim=1) # (batch_size, num_patches + 1, embed_dim)
        x = x + self.positional_embedding
        x = self.dropout1d(x)
        x = self.transformer(x) # (batch_size, num_patches + 1, embed_dim)
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
            specification = kwargs.get("specification") or "B/32"

            in_channels = 3
            image_size = 224

            enc_dropout, dropout = 0, 0
            activation = nn.GELU()
            pooling = "avg"
            bias_head = True
            num_classes = 1000

            if specification[0] == "B":
                embed_dim = 768
                d_ff = 3072
                nhead, num_layers = 12, 12
            elif specification[0] == "L":
                embed_dim = 1024
                d_ff = 4096
                nhead, num_layers = 16, 24
            elif specification[0] == "H":
                embed_dim = 1280
                d_ff = 5120
                nhead, num_layers = 16, 32
            else:
                raise ValueError("Not support {}/*.".format(specification[0]))

            patch_size = int(specification[2:])
        else:
            raise ValueError("Not support task={}.".format(task))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, dim_feedforward=d_ff,
            dropout=enc_dropout,
            activation=activation,
            layer_norm_eps=EPS,
            batch_first=True, norm_first=True
        )
        transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers, norm=None)

        model = cls(
            transformer,
            in_channels=in_channels, embed_dim=embed_dim,
            image_size=image_size, patch_size=patch_size,
            dropout=dropout,
            pooling=pooling,
            bias_head=bias_head,
            num_classes=num_classes
        )

        return model

class ViTPool(nn.Module):
    def __init__(self, pooling="cls", dim=1):
        super().__init__()

        self.pooling = pooling
        self.dim = dim

        if not self.pooling in ["cls", "avg"]:
            raise ValueError("Not support pooling={}".format(self.pooling))

    def forward(self, input):
        dim = self.dim

        if self.pooling == "cls":
            sections = [1, input.size(dim) - 1]
            output, _ = torch.split(input, sections, dim=dim)
            output = output.squeeze(dim=dim)
        else:
            output = input.mean(dim=dim)

        return output

def _test_vit():
    patch_embed_dim = 768
    nhead, num_layers = 12, 12
    d_ff = 3072
    enc_dropout = 0

    enc_layer = nn.TransformerEncoderLayer(
        d_model=patch_embed_dim, nhead=nhead, dim_feedforward=d_ff,
        dropout=enc_dropout,
        activation=nn.GELU(),
        layer_norm_eps=EPS,
        batch_first=True, norm_first=True
    )
    transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers, norm=None)

    in_channels = 3
    image_size, patch_size = 224, 32
    dropout = 0
    pooling = "cls"
    bias_head = True
    num_classes = 1000

    model = ViT(
        transformer,
        in_channels=in_channels, embed_dim=patch_embed_dim,
        image_size=image_size, patch_size=patch_size,
        dropout=dropout,
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

    print("="*10, "Vision Transforner (ViT)", "="*10)
    _test_vit()