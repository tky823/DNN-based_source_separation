import torch
import torch.nn as nn

from utils.utils_tasnet import choose_layer_norm
from models.dprnn import IntraChunkRNN as LocallyRecurrentBlock

EPS=1e-12

class GALR(nn.Module):
    def __init__(self, num_features, hidden_channels, num_blocks=6, num_heads=4, causal=False, norm=True, eps=EPS):
        super().__init__()
        
        # Network confguration
        net = []
        
        for _ in range(num_blocks):
            net.append(GALRBlock(num_features, hidden_channels, num_heads=num_heads, causal=causal, norm=norm, eps=eps))
            
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
    def __init__(self, num_features, hidden_channels, num_heads, causal, norm=True, eps=EPS):
        super().__init__()
        
        self.intra_chunk_block = LocallyRecurrentBlock(num_features, hidden_channels=hidden_channels, norm=norm, eps=eps)
        self.inter_chunk_block = GloballyAttentiveBlock(num_features, num_heads=num_heads, causal=causal, norm=norm, eps=eps)
        
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

class GloballyAttentiveBlock(nn.Module):
    def __init__(self, num_features, num_heads, causal=False, norm=True, dropout=0.1, eps=EPS):
        super().__init__()

        self.norm = norm

        if self.norm:    
            self.norm1d_in = choose_layer_norm(num_features, causal=causal, eps=eps)

        self.multihead_attn = nn.MultiheadAttention(num_features, num_heads)
        if dropout is not None:
            self.dropout = True
            self.dropout1d = nn.Dropout(p=dropout)
        else:
            self.dropout = False
        if self.norm:    
            self.norm1d_out = choose_layer_norm(num_features, causal=causal, eps=eps)
        
    def forward(self, input):
        """
        Args:
            input (batch_size, num_features, S, chunk_size)
        Returns:
            output (batch_size, num_features, S, chunk_size)
        """
        batch_size, num_features, S, chunk_size = input.size()

        input = input.view(batch_size, num_features, S*chunk_size) # -> (batch_size, num_features, S*chunk_size)

        if self.norm:
            x = self.norm1d_in(input) # -> (batch_size, num_features, S*chunk_size)
        else:
            x = input
        encoding = self.positional_encoding(length=S*chunk_size, dimension=num_features).permute(1,0).to(x.device)
        x = x + encoding # -> (batch_size, num_features, S*chunk_size)
        x = x.view(batch_size, num_features, S, chunk_size)
        x = x.permute(2,0,3,1).contiguous() # -> (S, batch_size, chunk_size, num_features)
        x = x.view(S, batch_size*chunk_size, num_features) # -> (S, batch_size*chunk_size, num_features)

        residual = x # (S, batch_size*chunk_size, num_features)
        x, _ = self.multihead_attn(x, x, x) # (T_tgt, batch_size, num_features), (batch_size, T_tgt, T_src), where T_tgt = T_src = T
        if self.dropout:
            x = self.dropout1d(x)
        x = x + residual # -> (S, batch_size*chunk_size, num_features)
        x = x.view(S, batch_size, chunk_size, num_features)
        x = x.permute(1,3,0,2).contiguous() # -> (batch_size, num_features, S, chunk_size)
        x = x.view(batch_size, num_features, S*chunk_size) # -> (batch_size, num_features, S*chunk_size)
        if self.norm:
            x = self.norm1d_out(x) # -> (batch_size, num_features, S*chunk_size)
        x = x + input
        output = x.view(batch_size, num_features, S, chunk_size)

        return output
    
    def positional_encoding(self, length: int, dimension: int, base=10000):
        """
        Args:
            length <int>: 
            dimension <int>: 
        Returns:
            output (length, dimension): positional encording
        """
        assert dimension%2 == 0, "dimension is expected even number but given odd number."

        position = torch.arange(length) # (length,)
        position = position.unsqueeze(dim=1) # (length, 1)
        index = torch.arange(dimension//2) / dimension # (dimension//2,)
        index = index.unsqueeze(dim=0) # (1, dimension//2)
        indices = position / base**index
        output = torch.cat([torch.sin(indices), torch.cos(indices)], dim=1)
        
        return output

def _test_globally_attentive_block():
    batch_size = 4
    num_heads = 4
    num_features, chunk_size, S = 16, 10, 5

    input = torch.randint(0, 10, (batch_size, num_features, S, chunk_size), dtype=torch.float)

    globally_attentive_block = GloballyAttentiveBlock(num_features, num_heads=num_heads)
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
    print('-'*10, "Causal", '-'*10)
    causal = True
    
    model = GALR(num_features, hidden_channels, num_blocks=num_blocks, causal=causal)
    print(model)
    output = model(input)
    print(input.size(), output.size())
    print()
    
    # Non causal
    print('-'*10, "Non causal", '-'*10)
    causal = False
    
    model = GALR(num_features, hidden_channels, num_blocks=num_blocks, causal=causal)
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
