import torch
import torch.nn as nn
import torch.nn.functional as F

EPS=1e-12

class MulCatDPRNN(nn.Module):
    def __init__(self, num_features, hidden_channels, num_blocks=6, causal=False, n_sources=2, eps=EPS):
        super().__init__()

        assert num_blocks % 2 == 0, "num_blocks is expected to be even."

        self.num_blocks = num_blocks
        self.n_sources = n_sources
        
        # Network confguration
        net = []
        
        for idx in range(num_blocks):
            if idx % 2 == 0:
                net.append(MulCatIntraChunkRNN(num_features, hidden_channels, eps=eps))
            else:
                net.append(MulCatInterChunkRNN(num_features, hidden_channels, causal=causal, eps=eps))
            
        self.net = nn.Sequential(*net)
        self.prelu = nn.PReLU()
        self.pointwise_conv2d = nn.Conv2d(num_features, n_sources*num_features, kernel_size=(1,1), stride=(1,1))

    def forward(self, input):
        """
        Args:
            input (batch_size, num_features, S, chunk_size)
        Returns:
            output (batch_size, num_blocks//2, n_sources, num_features, S, chunk_size)
        """
        num_blocks = self.num_blocks
        n_sources = self.n_sources

        output = []

        batch_size, num_features, S, chunk_size = input.size()

        x = input

        for idx in range(num_blocks):
            x = self.net[idx](x)
            if idx % 2 == 0:
                x_output = self.prelu(x)
                x_output = self.pointwise_conv2d(x_output) # (batch_size, num_features, S, chunk_size) -> (batch_size, n_sources*num_features, S, chunk_size)
                x_output = x_output.view(batch_size, 1, n_sources, num_features, S, chunk_size) # -> (batch_size, 1, n_sources, num_features, S, chunk_size)
                output.append(x_output)

        output = torch.cat(output, dim=1) # -> (batch_size, num_blocks//2, n_sources, num_features, S, chunk_size)

        return output

class MulCatIntraChunkRNN(nn.Module):
    def __init__(self, num_features, hidden_channels, eps=EPS):
        super().__init__()
        
        self.num_features, self.hidden_channels = num_features, hidden_channels
        num_directions = 2 # bi-direction
        
        self.rnn_output = nn.LSTM(num_features, hidden_channels//num_directions, batch_first=True, bidirectional=True)
        self.rnn_gate = nn.LSTM(num_features, hidden_channels//num_directions, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_channels + num_features, num_features)
        
    def forward(self, input):
        """
        Args:
            input (batch_size, num_features, S, chunk_size)
        Returns:
            output (batch_size, num_features, S, chunk_size)
        """
        num_features, hidden_channels = self.num_features, self.hidden_channels
        batch_size, _, S, chunk_size = input.size()
        
        residual = input
        x_input = input.permute(0,2,3,1).contiguous() # (batch_size, num_features, S, chunk_size) -> (batch_size, S, chunk_size, num_features)
        x_input = x_input.view(batch_size*S, chunk_size, num_features) # (batch_size*S, chunk_size, num_features)
        x_output, (_, _) = self.rnn_output(x_input) # (batch_size*S, chunk_size, num_features) -> (batch_size*S, chunk_size, hidden_channels)
        x_gate, (_, _) = self.rnn_gate(x_input) # (batch_size*S, chunk_size, num_features) -> (batch_size*S, chunk_size, hidden_channels)
        x = x_output * x_gate # (batch_size*S, chunk_size, hidden_channels)
        x = torch.cat([x, x_input], dim=2) # -> (batch_size*S, chunk_size, hidden_channels + num_features)
        x = self.fc(x) # -> (batch_size*S, chunk_size, num_features)
        x = x.permute(0,2,1) # -> (batch_size*S, num_features, chunk_size)
        x = x.view(batch_size, S, num_features, chunk_size) # (batch_size, S, num_features, chunk_size)
        x = x.permute(0,2,1,3) # (batch_size, num_features, S, chunk_size)
        output = x + residual
        
        return output

class MulCatInterChunkRNN(nn.Module):
    def __init__(self, num_features, hidden_channels, causal, eps=EPS):
        super().__init__()
        
        self.num_features, self.hidden_channels = num_features, hidden_channels
        
        if causal: # uni-direction
            self.rnn_output = nn.LSTM(num_features, hidden_channels, batch_first=True, bidirectional=False)
            self.rnn_gate = nn.LSTM(num_features, hidden_channels, batch_first=True, bidirectional=False)
        else: # bi-direction
            self.rnn_output = nn.LSTM(num_features, hidden_channels//2, batch_first=True, bidirectional=True)
            self.rnn_gate = nn.LSTM(num_features, hidden_channels//2, batch_first=True, bidirectional=True)
        
        self.fc = nn.Linear(hidden_channels + num_features, num_features)
        
    def forward(self, input):
        """
        Args:
            input (batch_size, num_features, S, chunk_size)
        Returns:
            output (batch_size, num_features, S, chunk_size)
        """
        num_features, hidden_channels = self.num_features, self.hidden_channels
        batch_size, _, S, chunk_size = input.size()
        
        residual = input
        x_input = input.permute(0,3,2,1).contiguous() # (batch_size, num_features, S, chunk_size) -> (batch_size, chunk_size, S, num_features)
        x_input = x_input.view(batch_size*chunk_size, S, num_features) # -> (batch_size*chunk_size, S, num_features)
        x_output, (_, _) = self.rnn_output(x_input) # -> (batch_size*chunk_size, S, hidden_channels)
        x_gate, (_, _) = self.rnn_gate(x_input) # -> (batch_size*chunk_size, S, hidden_channels)
        x = x_output * x_gate # (batch_size*chunk_size, S, hidden_channels)
        x = torch.cat([x, x_input], dim=2) # -> (batch_size*chunk_size, S, hidden_channels + num_features)
        x = self.fc(x) # -> (batch_size*chunk_size, S, num_features)
        x = x.permute(0,2,1) # -> (batch_size*chunk_size, num_features, S)
        x = x.view(batch_size, chunk_size, num_features, S) # -> (batch_size, chunk_size, num_features, S)
        x = x.permute(0,2,3,1).contiguous() # -> (batch_size, num_features, S, chunk_size)
        
        output = x + residual
        
        return output

if __name__ == '__main__':
    batch_size = 4
    S, chunk_size = 5, 3
    num_features, hidden_channels = 8, 32
    num_blocks = 6
    causal = False
    n_sources = 2

    input = torch.randint(0, 10, (batch_size, num_features, S, chunk_size), dtype=torch.float)

    model = MulCatDPRNN(num_features, hidden_channels, num_blocks=num_blocks, causal=causal, n_sources=n_sources)
    print(model)

    output = model(input)
    print(input.size(), output.size())
    print()