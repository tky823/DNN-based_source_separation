import warnings

import torch
import torch.nn as nn

from utils.tasnet import choose_layer_norm

EPS = 1e-12

class DualPathTransformer(nn.Module):
    def __init__(self, num_features, hidden_channels, num_blocks=6, num_heads=4, norm=True, nonlinear='relu', dropout=0, causal=False, eps=EPS):
        super().__init__()

        warnings.warn("Use models.dptnet.DualPathTransformer instead.", DeprecationWarning)

        # Network confguration
        net = []

        for _ in range(num_blocks):
            net.append(DualPathTransformerBlock(num_features, hidden_channels, num_heads=num_heads, norm=norm, nonlinear=nonlinear, dropout=dropout, causal=causal, eps=eps))

        self.net = nn.Sequential(*net)

    def forward(self, input):
        """
        Args:
            input (batch_size, num_features, S, chunk_size)
        Returns:
            output (batch_size, num_features, S, chunk_size)
        """
        output = self.net(input)

        return output

class DualPathTransformerBlock(nn.Module):
    def __init__(self, num_features, hidden_channels, num_heads=4, norm=True, nonlinear='relu', dropout=0, causal=False, eps=EPS):
        super().__init__()

        warnings.warn("Use models.dptnet.DualPathTransformerBlock instead.", DeprecationWarning)

        self.intra_chunk_block = IntraChunkTransformer(
            num_features, hidden_channels, num_heads=num_heads,
            norm=norm, nonlinear=nonlinear, dropout=dropout,
            eps=eps
        )
        self.inter_chunk_block = InterChunkTransformer(
            num_features, hidden_channels, num_heads=num_heads,
            norm=norm, nonlinear=nonlinear, dropout=dropout,
            causal=causal,
            eps=eps
        )

    def forward(self, input):
        """
        Args:
            input (batch_size, num_features, S, chunk_size)
        Returns:
            output (batch_size, num_features, S, chunk_size)
        """
        x = self.intra_chunk_block(input)
        output = self.inter_chunk_block(x)

        return output

class IntraChunkTransformer(nn.Module):
    def __init__(self, num_features, hidden_channels, num_heads=4, norm=True, nonlinear='relu', dropout=0, eps=EPS):
        super().__init__()

        warnings.warn("Use models.dptnet.IntraChunkTransformer instead.", DeprecationWarning)

        self.num_features = num_features

        self.transformer = ImprovedTransformer(
            num_features, hidden_channels, num_heads=num_heads,
            norm=norm, nonlinear=nonlinear, dropout=dropout,
            causal=False,
            eps=eps
        )

    def forward(self, input):
        """
        Args:
            input (batch_size, num_features, S, chunk_size)
        Returns:
            output (batch_size, num_features, S, chunk_size)
        """
        num_features = self.num_features
        batch_size, _, S, chunk_size = input.size()

        x = input.permute(3, 0, 2, 1).contiguous() # (batch_size, num_features, S, chunk_size) -> (chunk_size, batch_size, S, num_features)
        x = x.view(chunk_size, batch_size*S, num_features) # (chunk_size, batch_size, S, num_features) -> (chunk_size, batch_size*S, num_features)
        x = self.transformer(x) # -> (chunk_size, batch_size*S, num_features)
        x = x.view(chunk_size, batch_size, S, num_features) # (chunk_size, batch_size*S, num_features) -> (chunk_size, batch_size, S, num_features)
        output = x.permute(1, 3, 2, 0) # (chunk_size, batch_size, S, num_features) -> (batch_size, num_features, S, chunk_size)

        return output

class InterChunkTransformer(nn.Module):
    def __init__(self, num_features, hidden_channels, num_heads=4, causal=False, norm=True, nonlinear='relu', dropout=0, eps=EPS):
        super().__init__()

        warnings.warn("Use models.dptnet.InterChunkTransformer instead.", DeprecationWarning)

        self.num_features = num_features

        self.transformer = ImprovedTransformer(
            num_features, hidden_channels, num_heads=num_heads,
            norm=norm, nonlinear=nonlinear, dropout=dropout,
            causal=causal,
            eps=eps
        )

    def forward(self, input):
        """
        Args:
            input (batch_size, num_features, S, chunk_size)
        Returns:
            output (batch_size, num_features, S, chunk_size)
        """
        num_features = self.num_features
        batch_size, _, S, chunk_size = input.size()

        x = input.permute(2, 0, 3, 1).contiguous() # (batch_size, num_features, S, chunk_size) -> (S, batch_size, chunk_size, num_features)
        x = x.view(S, batch_size*chunk_size, num_features) # (S, batch_size, chunk_size, num_features) -> (S, batch_size*chunk_size, num_features)
        x = self.transformer(x) # -> (S, batch_size*chunk_size, num_features)
        x = x.view(S, batch_size, chunk_size, num_features) # (S, batch_size*chunk_size, num_features) -> (S, batch_size, chunk_size, num_features)
        output = x.permute(1, 3, 0, 2) # (S, batch_size, chunk_size, num_features) -> (batch_size, num_features, S, chunk_size)

        return output

class ImprovedTransformer(nn.Module):
    def __init__(self, num_features, hidden_channels, num_heads=4, norm=True, nonlinear='relu', dropout=0, causal=False, eps=EPS):
        super().__init__()

        warnings.warn("Use models.dptnet.ImprovedTransformer instead.", DeprecationWarning)

        self.multihead_attn_block = MultiheadAttentionBlock(num_features, num_heads, norm=norm, dropout=dropout, causal=causal, eps=eps)
        self.subnet = FeedForwardBlock(num_features, hidden_channels, norm=norm, nonlinear=nonlinear, causal=causal, eps=eps)

    def forward(self, input):
        """
        Args:
            input (T, batch_size, num_features)
        Returns:
            output (T, batch_size, num_features)
        """
        x = self.multihead_attn_block(input)
        output = self.subnet(x)

        return output

class MultiheadAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, norm=True, dropout=0, causal=False, eps=EPS):
        super().__init__()

        warnings.warn("Use models.dptnet.MultiheadAttentionBlock instead.", DeprecationWarning)

        if dropout == 0:
            self.dropout = False
        else:
            self.dropout = True

        self.norm = norm

        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)

        if self.dropout:
            self.dropout1d = nn.Dropout(p=dropout)
        if self.norm:
            norm_name = 'cLN' if causal else 'gLM'
            self.norm1d = choose_layer_norm(norm_name, embed_dim, causal=causal, eps=eps)

    def forward(self, input):
        """
        Args:
            input (T, batch_size, embed_dim)
        Returns:
            output (T, batch_size, embed_dim)
        """
        x = input # (T, batch_size, embed_dim)

        residual = x
        x, _ = self.multihead_attn(x, x, x) # (T_tgt, batch_size, embed_dim), (batch_size, T_tgt, T_src), where T_tgt = T_src = T
        x = x + residual

        if self.dropout:
            x = self.dropout1d(x)

        if self.norm:
            x = x.permute(1, 2, 0) # (batch_size, embed_dim, T)
            x = self.norm1d(x) # (batch_size, embed_dim, T)
            x = x.permute(2, 0, 1).contiguous() # (batch_size, embed_dim, T) -> (T, batch_size, embed_dim)

        output = x

        return output

class FeedForwardBlock(nn.Module):
    def __init__(self, num_features, hidden_channels, norm=True, nonlinear='relu', causal=False, eps=EPS):
        super().__init__()

        warnings.warn("Use models.dptnet.FeedForwardBlock instead.", DeprecationWarning)

        if causal:
            bidirectional = False
            num_directions = 1 # uni-direction
        else:
            bidirectional = True
            num_directions = 2 # bi-direction

        self.norm = norm

        self.rnn = nn.LSTM(num_features, hidden_channels, batch_first=False, bidirectional=bidirectional)
        if nonlinear == 'relu':
            self.nonlinear1d = nn.ReLU()
        else:
            raise ValueError("Not support nonlinear function {}.".format(nonlinear))
        self.fc = nn.Linear(num_directions*hidden_channels, num_features)

        if self.norm:
            norm_name = 'cLN' if causal else 'gLM'
            self.norm1d = choose_layer_norm(norm_name, num_features, causal=causal, eps=eps)

    def forward(self, input):
        """
        Args:
            input (T, batch_size, num_features)
        Returns:
            output (T, batch_size, num_features)
        """
        x = input # (T, batch_size, num_features)

        self.rnn.flatten_parameters()

        residual = x
        x, _ = self.rnn(x) # (T, batch_size, num_features) -> (T, batch_size, num_directions*hidden_channels)
        x = self.nonlinear1d(x) # -> (T, batch_size, num_directions*hidden_channels)
        x = self.fc(x) # (T, batch_size, num_directions*hidden_channels) -> (T, batch_size, num_features)
        x = x + residual

        if self.norm:
            x = x.permute(1, 2, 0) # (T, batch_size, num_features) -> (batch_size, num_features, T)
            x = self.norm1d(x) # (batch_size, num_features, T)
            x = x.permute(2, 0, 1).contiguous() # (batch_size, num_features, T) -> (T, batch_size, num_features)

        output = x

        return output

def _test_multihead_attn_block():
    batch_size = 2
    T = 10
    embed_dim = 8
    num_heads = 4
    input = torch.randn((T, batch_size, embed_dim), dtype=torch.float)

    print('-'*10, "Non causal & No dropout", '-'*10)
    causal = False
    dropout = 0

    model = MultiheadAttentionBlock(embed_dim, num_heads=num_heads, dropout=dropout, causal=causal)
    print(model)

    output = model(input)
    print(input.size(), output.size())
    print()

    print('-'*10, "Causal & Dropout (p=0.3)", '-'*10)
    causal = True
    dropout = 0.3

    model = MultiheadAttentionBlock(embed_dim, num_heads=num_heads, dropout=dropout, causal=causal)
    print(model)

    output = model(input)
    print(input.size(), output.size())

def _test_feedforward_block():
    batch_size = 2
    T = 10
    num_features, hidden_channels = 12, 10

    input = torch.randn((T, batch_size, num_features), dtype=torch.float)

    print('-'*10, "Causal", '-'*10)
    causal = True
    nonlinear = 'relu'

    model = FeedForwardBlock(num_features, hidden_channels, nonlinear=nonlinear, causal=causal)
    print(model)

    output = model(input)
    print(input.size(), output.size())

def _test_improved_transformer():
    batch_size = 2
    T = 10
    num_features, hidden_channels = 12, 10
    num_heads = 4

    input = torch.randn((T, batch_size, num_features), dtype=torch.float)

    print('-'*10, "Non causal", '-'*10)
    causal = False

    model = ImprovedTransformer(num_features, hidden_channels, num_heads=num_heads, causal=causal)
    print(model)

    output = model(input)
    print(input.size(), output.size())

def _test_transformer_block():
    batch_size = 2
    num_features, hidden_channels = 12, 8
    S, chunk_size = 10, 5 # global length and local length
    num_heads = 3
    input = torch.randn((batch_size, num_features, S, chunk_size), dtype=torch.float)

    print('-'*10, "transformer block for intra chunk", '-'*10)
    model = IntraChunkTransformer(num_features, hidden_channels=hidden_channels, num_heads=num_heads)
    print(model)

    output = model(input)
    print(input.size(), output.size())
    print()

    print('-'*10, "transformer block for inter chunk", '-'*10)
    causal = True
    model = InterChunkTransformer(num_features, hidden_channels=hidden_channels, num_heads=num_heads, causal=causal)
    print(model)

    output = model(input)
    print(input.size(), output.size())

def _test_dptransformer():
    batch_size = 2
    num_features, hidden_channels = 12, 8
    S, chunk_size = 10, 5 # global length and local length
    num_blocks = 6
    num_heads = 3
    input = torch.randn((batch_size, num_features, S, chunk_size), dtype=torch.float)
    causal = True

    model = DualPathTransformer(num_features, hidden_channels, num_blocks=num_blocks, num_heads=num_heads, causal=causal)
    print(model)

    output = model(input)
    print(input.size(), output.size())

if __name__ == '__main__':
    print('='*10, "Multihead attention block", '='*10)
    _test_multihead_attn_block()
    print()

    print('='*10, "feed-forward block", '='*10)
    _test_feedforward_block()
    print()

    print('='*10, "improved transformer", '='*10)
    _test_improved_transformer()
    print()

    print('='*10, "transformer block", '='*10)
    _test_transformer_block()
    print()

    print('='*10, "Dual path transformer network", '='*10)
    _test_dptransformer()
    print()