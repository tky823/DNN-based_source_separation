import torch
import torch.nn as nn

from utils.utils_tasnet import choose_layer_norm

EPS=1e-12

class DPRNN(nn.Module):
    def __init__(self, num_features, hidden_channels, num_blocks=6, causal=False):
        super().__init__()
        
        # Network confguration
        net = []
        
        for _ in range(num_blocks):
            net.append(DPRNNBlock(num_features, hidden_channels, causal=causal))
            
        self.net = nn.Sequential(*net)

    def forward(self, input):
        """
        Args:
            input (batch_size, num_features, chunk_size, S)
        Returns:
            input (batch_size, num_features, chunk_size, S)
        """
        output = self.net(input)

        return output

class DPRNNBlock(nn.Module):
    def __init__(self, num_features, hidden_channels, causal):
        super().__init__()
        
        self.intra_chunk_block = IntraChunkRNN(num_features, hidden_channels)
        self.inter_chunk_block = InterChunkRNN(num_features, hidden_channels, causal=causal)
        
    def forward(self, input):
        """
        Args:
            input (batch_size, num_features, S, chunk_size)
        """
        x = self.intra_chunk_block(input)
        output = self.inter_chunk_block(x)
        
        return output

class IntraChunkRNN(nn.Module):
    def __init__(self, num_features, hidden_channels, eps=EPS):
        super().__init__()
        
        self.num_features, self.hidden_channels = num_features, hidden_channels
        num_directions = 2 # bi-direction
        
        self.rnn = nn.LSTM(num_features, hidden_channels//num_directions, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_channels, num_features)
        self.norm1d = choose_layer_norm(num_features, causal=False, eps=eps)
        
    def forward(self, input):
        """
        Args:
            input (batch_size, num_features, S, chunk_size)
        """
        num_features, hidden_channels = self.num_features, self.hidden_channels
        batch_size, _, S, chunk_size = input.size()
        
        residual = input
        x = input.permute(0,2,3,1).contiguous() # (batch_size, num_features, S, chunk_size) -> (batch_size, S, chunk_size, num_features)
        x = x.view(batch_size*S, chunk_size, num_features)
        x, (_, _) = self.rnn(x) # (batch_size*S, chunk_size, num_features) -> (batch_size*S, chunk_size, hidden_channels)
        x = self.fc(x) # -> (batch_size*S, chunk_size, num_features)
        x = x.permute(0,2,1) # -> (batch_size*S, num_features, chunk_size)
        x = self.norm1d(x) # (batch_size*S, num_features, chunk_size)
        x = x.view(batch_size, S, num_features, chunk_size) # (batch_size, S, num_features, chunk_size)
        x = x.permute(0,2,1,3) # (batch_size, num_features, S, chunk_size)
        output = x + residual
        
        return output

class InterChunkRNN(nn.Module):
    def __init__(self, num_features, hidden_channels, causal, eps=EPS):
        super().__init__()
        
        self.num_features, self.hidden_channels = num_features, hidden_channels
        
        if causal: # uni-direction
            self.rnn = nn.LSTM(num_features, hidden_channels, batch_first=True, bidirectional=False)
        else: # bi-direction
            self.rnn = nn.LSTM(num_features, hidden_channels//2, batch_first=True, bidirectional=True)
        
        self.fc = nn.Linear(hidden_channels, num_features)
        self.norm1d = choose_layer_norm(num_features, causal=False, eps=eps)
        
    def forward(self, input):
        """
        Args:
            input (batch_size, num_features, S, chunk_size)
        """
        num_features, hidden_channels = self.num_features, self.hidden_channels
        batch_size, _, S, chunk_size = input.size()
        
        residual = input
        x = input.permute(0,3,2,1).contiguous() # (batch_size, num_features, S, chunk_size) -> (batch_size, chunk_size, S, num_features)
        x = x.view(batch_size*chunk_size, S, num_features) # -> (batch_size*chunk_size, S, num_features)
        x, (_, _) = self.rnn(x) # -> (batch_size*chunk_size, S, hidden_channels)
        x = self.fc(x) # -> (batch_size*chunk_size, S, num_features)
        x = x.permute(0,2,1) # -> (batch_size*chunk_size, num_features, S)
        x = self.norm1d(x) # -> (batch_size*chunk_size, num_features, S)
        x = x.view(batch_size, chunk_size, num_features, S) # -> (batch_size, chunk_size, num_features, S)
        x = x.permute(0,2,3,1).contiguous() # -> (batch_size, num_features, S, chunk_size)
        
        output = x + residual
        
        return output

if __name__ == '__main__':
    batch_size = 4
    num_features, chunk_size, S = 64, 10, 4
    hidden_channels = 32
    num_blocks = 3
    
    input = torch.randint(0, 10, (batch_size, num_features, S, chunk_size), dtype=torch.float)

    # Causal
    causal = True
    
    model = DPRNN(num_features, hidden_channels, num_blocks=num_blocks, causal=causal)
    print(model)
    output = model(input)
    print(input.size(), output.size())
    
    # Non causal
    causal = False
    
    model = DPRNN(num_features, hidden_channels, num_blocks=num_blocks, causal=causal)
    print(model)
    output = model(input)
    print(input.size(), output.size())
