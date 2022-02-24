import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair

from models.transform import SplitToPatch

class ViT(nn.Module):
    """
    Vision Transformer
    Reference:
        "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
        See https://arxiv.org/abs/2010.11929
    """
    def __init__(self, transformer, in_channels, embed_dim, image_size, patch_size=16, dropout=0, emb_dropout=0, pooling="cls", num_classes=1000, eps=1e-12):
        super().__init__()

        image_size = _pair(image_size)
        patch_size = _pair(patch_size)

        H, W = image_size
        pH, pW = patch_size

        num_patches = (H // pH) * (W // pW)

        self.split_to_patch = SplitToPatch(patch_size, channel_first=False)
        self.embedding = nn.Linear(in_channels * pH * pW, embed_dim)
        self.dropout = nn.Dropout(p=emb_dropout)
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
        x = self.dropout(x)
        x = self.transformer(x) # (batch_size, num_patches + 1, embed_dim)
        x = self.pooling2d(x) # (batch_size, embed_dim)
        x = self.norm2d(x) # (batch_size, embed_dim)
        output = self.fc(x) # (batch_size, num_classes)

        return output

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