import torch
import torch.nn as nn

from utils.tasnet import choose_layer_norm
from models.dprnn import IntraChunkRNN as LocallyRecurrentBlock

EPS = 1e-12

class GALR(nn.Module):
    def __init__(self, num_features, hidden_channels, num_blocks=6, num_heads=8, norm=True, dropout=1e-1, low_dimension=True, causal=False, eps=EPS, **kwargs):
        super().__init__()

        # Network confguration
        net = []

        for _ in range(num_blocks):
            net.append(GALRBlock(num_features, hidden_channels, num_heads=num_heads, norm=norm, dropout=dropout, low_dimension=low_dimension, causal=causal, eps=eps, **kwargs))

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

class GALRBlock(nn.Module):
    def __init__(self, num_features, hidden_channels, num_heads=8, causal=False, norm=True, dropout=1e-1, low_dimension=True, eps=EPS, **kwargs):
        super().__init__()

        self.intra_chunk_block = LocallyRecurrentBlock(num_features, hidden_channels=hidden_channels, norm=norm, eps=eps)

        if low_dimension:
            chunk_size = kwargs['chunk_size']
            down_chunk_size = kwargs['down_chunk_size']
            self.inter_chunk_block = LowDimensionGloballyAttentiveBlock(num_features, chunk_size=chunk_size, down_chunk_size=down_chunk_size, num_heads=num_heads, causal=causal, norm=norm, dropout=dropout, eps=eps)
        else:
            self.inter_chunk_block = GloballyAttentiveBlock(num_features, num_heads=num_heads, causal=causal, norm=norm, dropout=dropout, eps=eps)

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

class GloballyAttentiveBlockBase(nn.Module):
    def __init__(self):
        super().__init__()

    def positional_encoding(self, length: int, dimension: int, base=10000):
        """
        Args:
            length <int>: 
            dimension <int>: 
        Returns:
            output (length, dimension): positional encording
        """
        assert dimension % 2 == 0, "dimension is expected even number but given odd number."

        position = torch.arange(length) # (length,)
        position = position.unsqueeze(dim=1) # (length, 1)
        index = torch.arange(dimension//2) / dimension # (dimension // 2,)
        index = index.unsqueeze(dim=0) # (1, dimension // 2)
        indices = position / base**index
        output = torch.cat([torch.sin(indices), torch.cos(indices)], dim=1)

        return output

class GloballyAttentiveBlock(GloballyAttentiveBlockBase):
    def __init__(self, num_features, num_heads=8, causal=False, norm=True, dropout=1e-1, eps=EPS):
        super().__init__()

        self.norm = norm

        if self.norm:
            self.norm2d_in = LayerNormAlongChannel(num_features, eps=eps)

        self.multihead_attn = nn.MultiheadAttention(num_features, num_heads)

        if dropout is not None:
            self.dropout = True
            self.dropout1d = nn.Dropout(p=dropout)
        else:
            self.dropout = False

        if self.norm:
            norm_name = 'cLN' if causal else 'gLN'
            self.norm2d_out = choose_layer_norm(norm_name, num_features, causal=causal, eps=eps)

    def forward(self, input):
        """
        Args:
            input (batch_size, num_features, S, K): K is chunk size
        Returns:
            output (batch_size, num_features, S, K)
        """
        batch_size, num_features, S, K = input.size()

        if self.norm:
            x = self.norm2d_in(input) # -> (batch_size, num_features, S, K)
        else:
            x = input
        encoding = self.positional_encoding(length=S*K, dimension=num_features).permute(1,0).view(num_features, S, K).to(x.device)
        x = x + encoding # -> (batch_size, num_features, S, K)
        x = x.permute(2, 0, 3, 1).contiguous() # -> (S, batch_size, K, num_features)
        x = x.view(S, batch_size*K, num_features) # -> (S, batch_size*K, num_features)

        residual = x # (S, batch_size*K, num_features)
        x, _ = self.multihead_attn(x, x, x) # (T_tgt, batch_size, num_features), (batch_size, T_tgt, T_src), where T_tgt = T_src = T

        if self.dropout:
            x = self.dropout1d(x)
        x = x + residual # -> (S, batch_size*K, num_features)
        x = x.view(S, batch_size, K, num_features)
        x = x.permute(1, 3, 0, 2).contiguous() # -> (batch_size, num_features, S, K)

        if self.norm:
            x = self.norm2d_out(x) # -> (batch_size, num_features, S, K)
        x = x + input
        output = x.view(batch_size, num_features, S, K)

        return output

class LowDimensionGloballyAttentiveBlock(GloballyAttentiveBlockBase):
    def __init__(self, num_features, chunk_size=100, down_chunk_size=32, num_heads=8, causal=False, norm=True, dropout=1e-1, eps=EPS):
        super().__init__()

        self.down_chunk_size = down_chunk_size
        self.norm = norm

        self.fc_map = nn.Linear(chunk_size, down_chunk_size)

        if self.norm:
            self.norm2d_in = LayerNormAlongChannel(num_features, eps=eps)

        self.multihead_attn = nn.MultiheadAttention(num_features, num_heads)

        if dropout is not None:
            self.dropout = True
            self.dropout1d = nn.Dropout(p=dropout)
        else:
            self.dropout = False

        if self.norm:
            norm_name = 'cLN' if causal else 'gLN'
            self.norm2d_out = choose_layer_norm(norm_name, num_features, causal=causal, eps=eps)

        self.fc_inv = nn.Linear(down_chunk_size, chunk_size)

    def forward(self, input):
        """
        Args:
            input (batch_size, num_features, S, K): K is chunk size
        Returns:
            output (batch_size, num_features, S, K)
        """
        Q = self.down_chunk_size
        batch_size, num_features, S, K = input.size()

        x = self.fc_map(input) # (batch_size, num_features, S, K) -> (batch_size, num_features, S, Q)

        if self.norm:
            x = self.norm2d_in(x) # -> (batch_size, num_features, S, Q)

        encoding = self.positional_encoding(length=S*Q, dimension=num_features).permute(1,0).view(num_features, S, Q).to(x.device)
        x = x + encoding # -> (batch_size, num_features, S, Q)
        x = x.permute(2, 0, 3, 1).contiguous() # -> (S, batch_size, Q, num_features)
        x = x.view(S, batch_size*Q, num_features) # -> (S, batch_size*Q, num_features)

        residual = x # (S, batch_size*Q, num_features)
        x, _ = self.multihead_attn(x, x, x) # (T_tgt, batch_size, num_features), (batch_size, T_tgt, T_src), where T_tgt = T_src = T

        if self.dropout:
            x = self.dropout1d(x)
        x = x + residual # -> (S, batch_size*Q, num_features)
        x = x.view(S, batch_size, Q, num_features)
        x = x.permute(1, 3, 0, 2).contiguous() # -> (batch_size, num_features, S, Q)

        if self.norm:
            x = self.norm2d_out(x) # -> (batch_size, num_features, S, Q)

        x = self.fc_inv(x) # (batch_size, num_features, S, Q) -> (batch_size, num_features, S, K)
        x = x + input
        output = x.view(batch_size, num_features, S, K)

        return output

class LayerNormAlongChannel(nn.Module):
    def __init__(self, num_features, eps=EPS):
        super().__init__()

        self.num_features = num_features
        self.eps = eps

        self.norm = nn.LayerNorm(num_features, eps=eps)

    def forward(self, input):
        """
        Args:
            input (batch_size, num_features, *)
        Returns:
            output (batch_size, num_features, *)
        """
        n_dims = input.dim()
        dims = list(range(n_dims))
        permuted_dims = dims[0:1] + dims[2:] + dims[1:2]
        x = input.permute(*permuted_dims)
        x = self.norm(x)
        permuted_dims = dims[0:1] + dims[-1:] + dims[1:-1]
        output = x.permute(*permuted_dims).contiguous()

        return output

    def __repr__(self):
        s = '{}'.format(self.__class__.__name__)
        s += '({num_features}, eps={eps})'

        return s.format(**self.__dict__)

def _test_globally_attentive_block():
    batch_size = 4
    num_heads = 4
    num_features, chunk_size, S = 16, 10, 5

    input = torch.randint(0, 10, (batch_size, num_features, S, chunk_size), dtype=torch.float)

    print('-'*10, 'Non low dimension', '-'*10)
    globally_attentive_block = GloballyAttentiveBlock(num_features, num_heads=num_heads)
    print(globally_attentive_block)
    output = globally_attentive_block(input)
    print(input.size(), output.size())

    print('-'*10, 'Low dimension', '-'*10)
    down_chunk_size = 4
    globally_attentive_block = LowDimensionGloballyAttentiveBlock(num_features, chunk_size=chunk_size, down_chunk_size=down_chunk_size, num_heads=num_heads)
    print(globally_attentive_block)
    output = globally_attentive_block(input)
    print(input.size(), output.size())

def _test_galr():
    batch_size = 4
    num_features, chunk_size, S = 64, 10, 4
    hidden_channels = 32
    num_blocks = 3

    input = torch.randint(0, 10, (batch_size, num_features, S, chunk_size), dtype=torch.float)

    # Causal
    print('-'*10, "Causal and Non Low dimension", '-'*10)
    low_dimension = False
    causal = True

    model = GALR(num_features, hidden_channels, num_blocks=num_blocks, low_dimension=low_dimension, causal=causal)
    print(model)
    output = model(input)
    print(input.size(), output.size())
    print()

    # Non causal
    print('-'*10, "Non causal and Low dimension", '-'*10)
    low_dimension = True
    chunk_size, down_chunk_size = 10, 5
    causal = False

    model = GALR(num_features, hidden_channels, chunk_size=chunk_size, down_chunk_size=down_chunk_size, num_blocks=num_blocks, low_dimension=low_dimension, causal=causal)
    print(model)
    output = model(input)
    print(input.size(), output.size())

if __name__ == '__main__':
    print('='*10, "Globally attentive block", '='*10)
    _test_globally_attentive_block()
    print()

    print('='*10, "GALR", '='*10)
    _test_galr()
    print()
