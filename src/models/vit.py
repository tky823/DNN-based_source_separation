import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from models.transform import SplitToPatch

EPS = 1e-12

class ViT(nn.Module):
    """
    Vision Transformer
    Reference:
        "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
        See https://arxiv.org/abs/2010.11929
    """
    pretrained_model_ids = {
        "imagenet": {}
    }
    def __init__(
        self,
        transformer,
        in_channels, embed_dim,
        image_size, patch_size=16,
        dropout=0,
        pooling="cls",
        num_classes=1000,
        eps=EPS
    ):
        super().__init__()

        image_size = _pair(image_size)
        patch_size = _pair(patch_size)

        H, W = image_size
        pH, pW = patch_size

        num_patches = (H // pH) * (W // pW)

        self.split_to_patch = SplitToPatch(patch_size, channel_first=False)
        self.embedding = nn.Linear(in_channels * pH * pW, embed_dim)
        self.dropout1d = nn.Dropout(p=dropout)
        self.transformer = transformer
        self.pooling2d = Pooling(pooling, dim=1)
        self.norm2d = nn.LayerNorm(embed_dim, eps=eps)
        self.fc = nn.Linear(embed_dim, num_classes)

        self.pos_embedding = nn.Parameter(torch.empty(num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.empty(embed_dim))

        self._reset()

    def _reset(self):
        self.pos_embedding.data.normal_()
        self.cls_token.data.normal_()

    def forward(self, input):
        """
        Args:
            input: (batch_size, in_channels, height, width)
        Returns:
            output: (batch_size, num_classes)
        """
        x = self.split_to_patch(input)
        x = self.embedding(x) # (batch_size, num_patches, embed_dim)

        batch_size, _, embed_dim = x.size()

        cls_tokens = self.cls_token.view(1, 1, embed_dim)
        cls_tokens = cls_tokens.repeat(batch_size, 1, 1) # (batch_size, num_patches, embed_dim)

        x = torch.cat([cls_tokens, x], dim=1) # (batch_size, num_patches + 1, embed_dim)
        x = x + self.pos_embedding
        x = self.dropout1d(x)
        x = self.transformer(x) # (batch_size, num_patches + 1, embed_dim)
        x = self.pooling2d(x) # (batch_size, embed_dim)
        x = self.norm2d(x) # (batch_size, embed_dim)
        output = self.fc(x) # (batch_size, num_classes)

        return output

    @classmethod
    def build_from_pretrained(cls, load_state_dict=True, **kwargs):
        if load_state_dict:
            raise ValueError("Not support load_state_dict=True.")

        task = kwargs.get('task')

        if not task in cls.pretrained_model_ids:
            raise KeyError("Invalid task ({}) is specified.".format(task))

        if task == "imagenet":
            embed_dim = 1024
            nhead, num_layers = 16, 6
            d_ff = 2048

            enc_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=nhead, dim_feedforward=d_ff,
                activation=F.gelu,
                layer_norm_eps=EPS,
                batch_first=True, norm_first=True
            )
            transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

            in_channels = 3
            image_size, patch_size = 256, 16
            num_classes = 1000
        else:
            raise ValueError("Not support task={}.".format(task))

        model = cls(transformer, in_channels=in_channels, embed_dim=embed_dim, image_size=image_size, patch_size=patch_size, num_classes=num_classes)

        return model

class Pooling(nn.Module):
    def __init__(self, pooling="cls", dim=1):
        super().__init__()

        self.pooling = pooling
        self.dim = dim

        if not self.pooling in ["cls", "mean"]:
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
    in_channels = 3
    image_size, patch_size = 256, 16
    num_classes = 100

    embed_dim = 1024
    nhead, num_layers = 16, 6
    d_ff = 2048

    input = torch.randn(4, in_channels, image_size, image_size)

    enc_layer = nn.TransformerEncoderLayer(
        d_model=embed_dim, nhead=nhead, dim_feedforward=d_ff,
        activation=F.gelu,
        batch_first=True, norm_first=True
    )
    transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

    model = ViT(
        transformer,
        in_channels=in_channels, embed_dim=embed_dim,
        image_size=image_size, patch_size=patch_size,
        num_classes=num_classes
    )

    output = model(input)

    print(model)
    print(input.size(), output.size())

if __name__ == '__main__':
    torch.manual_seed(111)

    print("="*10, "Vision Transforner (ViT)", "="*10)
    _test_vit()