import torch
import torch.nn as nn

from utils.utils_tasnet import choose_layer_norm
from models.dprnn import IntraChunkRNN as LocallyRecurrentBlock

EPS=1e-12

class GALR(nn.Module):
    def __init__(self, num_features, hidden_channels, num_blocks=6, causal=False, norm=True, eps=EPS):
        super().__init__()
        
        # Network confguration
        net = []
        
        for _ in range(num_blocks):
            net.append(GALRBlock(num_features, hidden_channels, causal=causal, norm=norm, eps=eps))
            
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
    def __init__(self, num_features, hidden_channels, causal, norm=True, eps=EPS):
        super().__init__()
        
        self.intra_chunk_block = LocallyRecurrentBlock(num_features, hidden_channels, norm=norm, eps=eps)
        self.inter_chunk_block = GloballyAttentiveBlock(num_features, hidden_channels, causal=causal, norm=norm, eps=eps)
        
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
    def __init__(self, num_features, hidden_channels, causal, norm=True, eps=EPS):
        super().__init__()
        
        pass
        
    def forward(self, input):
        """
        Args:
            input (batch_size, num_features, S, chunk_size)
        Returns:
            output (batch_size, num_features, S, chunk_size)
        """
        pass

def _test_global_attentive_block():
    pass

def _test_galr():
    batch_size = 4
    num_features, chunk_size, S = 64, 10, 4
    hidden_channels = 32
    num_blocks = 3
    
    input = torch.randint(0, 10, (batch_size, num_features, S, chunk_size), dtype=torch.float)

    # Causal
    causal = True
    
    model = GALR(num_features, hidden_channels, num_blocks=num_blocks, causal=causal)
    print(model)
    output = model(input)
    print(input.size(), output.size())
    
    # Non causal
    causal = False
    
    model = GALR(num_features, hidden_channels, num_blocks=num_blocks, causal=causal)
    print(model)
    output = model(input)
    print(input.size(), output.size())


if __name__ == '__main__':
    _test_global_attentive_block()
    print()

    _test_galr()
    print()
