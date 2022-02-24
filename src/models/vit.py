import math

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
        self.pooling2d = ViTPool2d(pooling, dim=1)
        self.norm2d = nn.LayerNorm(embed_dim, eps=eps)
        self.fc = nn.Linear(embed_dim, num_classes)

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

        x = self.split_to_patch(input)
        x = self.embedding(x) # (batch_size, num_patches, embed_dim)

        cls_tokens = self.cls_token.repeat(batch_size, 1, 1) # (batch_size, num_patches, embed_dim)

        x = torch.cat([cls_tokens, x], dim=1) # (batch_size, num_patches + 1, embed_dim)
        x = x + self.positional_embedding
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
            d_model = 1024
            nhead, num_layers = 16, 6
            embed_dim = 6 * 64
            d_ff = 2048

            enc_layer = ViTEncoderLayer(
                d_model=d_model, nhead=nhead,
                embed_dim=embed_dim, dim_feedforward=d_ff,
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

class ViTPool2d(nn.Module):
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

class ViTEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model, nhead,
        embed_dim=384, dim_feedforward=1024,
        dropout=0, activation=F.gelu,
        layer_norm_eps=EPS,
        batch_first=True, norm_first=True,
        device=None, dtype=None
    ):
        super().__init__()

        factory_kwargs = {'device': device, 'dtype': dtype}

        self.self_attn = ViTMultiheadSelfAttention(embed_dim, qkv_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=batch_first, **factory_kwargs)
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def forward(self, input, src_mask=None, src_key_padding_mask=None):
        assert src_mask is None and src_key_padding_mask is None

        if self.norm_first:
            x = input + self._sa_block(self.norm1(input))
            output = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(input + self._sa_block(input))
            output = self.norm2(x + self._ff_block(x))

        return output

    def _sa_block(self, input):
        x = self.self_attn(input)
        output = self.dropout1(x)

        return output

    def _ff_block(self, input):
        x = self.linear1(input)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        output = self.dropout2(x)

        return output

class ViTMultiheadSelfAttention(nn.Module):
    def __init__(self, embed_dim, qkv_dim=None, num_heads=6, dropout=0, batch_first=True, device=None, dtype=None):
        super().__init__()

        factory_kwargs = {'device': device, 'dtype': dtype}

        self.embed_dim = embed_dim
        self.qkv_dim = qkv_dim = qkv_dim if qkv_dim is not None else embed_dim
        self.num_heads = num_heads
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj_weight = nn.Parameter(torch.empty((embed_dim, qkv_dim), **factory_kwargs), requires_grad=True)
        self.k_proj_weight = nn.Parameter(torch.empty((embed_dim, qkv_dim), **factory_kwargs), requires_grad=True)
        self.v_proj_weight = nn.Parameter(torch.empty((embed_dim, qkv_dim), **factory_kwargs), requires_grad=True)

        if embed_dim == qkv_dim:
            self.register_parameter("out_proj_weight", None)

            if dropout > 0:
                raise ValueError("dropout should be 0, when embed_dim == qkv_dim.")
        else:
            self.out_proj_weight = nn.Parameter(torch.empty((qkv_dim, embed_dim), **factory_kwargs), requires_grad=True)
            self.dropout1d = nn.Dropout(p=dropout)

    def forward(self, input):
        """
        Args:
            input:
                (batch_size, T, qkv_dim) if self.batch_first=True
                (T, batch_size, qkv_dim) if self.batch_first=False
            output:
                (batch_size, T, qkv_dim) if self.batch_first=True
                (T, batch_size, qkv_dim) if self.batch_first=False
        """
        num_heads = self.num_heads
        scale = math.sqrt(self.head_dim)

        if not self.batch_first:
            input = input.transpose(0, 1) # (T, batch_size, qkv_dim) -> (batch_size, T, qkv_dim)

        q = F.linear(input, self.q_proj_weight) # (batch_size, T, embed_dim)
        k = F.linear(input, self.k_proj_weight) # (batch_size, T, embed_dim)
        v = F.linear(input, self.v_proj_weight) # (batch_size, T, embed_dim)

        batch_size, T, embed_dim = q.size()

        q = q.view(batch_size, T, num_heads, embed_dim // num_heads)
        k = k.view(batch_size, T, num_heads, embed_dim // num_heads)
        v = v.view(batch_size, T, num_heads, embed_dim // num_heads)

        q = q.permute(0, 2, 1, 3) # (batch_size, num_heads, T, embed_dim // num_heads)
        k = k.permute(0, 2, 3, 1) # (batch_size, num_heads, embed_dim // num_heads, T)
        v = v.permute(0, 2, 1, 3) # (batch_size, num_heads, T, embed_dim // num_heads)

        qk = torch.matmul(q, k) / scale # (batch_size, num_heads, T, T)
        attn = F.softmax(qk, dim=-1) # (batch_size, num_heads, T, T)

        x = torch.matmul(attn, v) # (batch_size, num_heads, T, embed_dim // num_heads)
        x = x.permute(0, 2, 1, 3) # (batch_size, T, num_heads, embed_dim // num_heads)
        x = x.contiguous()
        x = x.view(batch_size, T, embed_dim) # (batch_size, T, embed_dim)

        if self.out_proj_weight is None:
            output = x
        else:
            x = F.linear(x, self.out_proj_weight) # (batch_size, T, embed_dim)
            output = self.dropout1d(x) # (batch_size, T, embed_dim)
        
        if not self.batch_first:
            output = output.transpose(0, 1) # (batch_size, T, qkv_dim) -> (T, batch_size, qkv_dim)

        return output
    
    def _reset_paramaters(self):
        nn.init.xavier_uniform_(self.q_proj_weight)
        nn.init.xavier_uniform_(self.k_proj_weight)
        nn.init.xavier_uniform_(self.v_proj_weight)

        if self.out_proj_weight is not None:
            nn.init.xavier_uniform_(self.out_proj_weight)

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


def _test_vit():
    in_channels = 3
    image_size, patch_size = 256, 16
    num_classes = 100

    d_model = 1024
    nhead, num_layers = 16, 6
    embed_dim = nhead * 64
    d_ff = 2048

    enc_layer = ViTEncoderLayer(
        d_model=d_model, nhead=nhead,
        embed_dim=embed_dim, dim_feedforward=d_ff,
        activation=F.gelu,
        layer_norm_eps=EPS,
        batch_first=True, norm_first=True
    )
    transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

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

    print("="*10, "Vision Transforner (ViT)", "="*10)
    _test_vit()