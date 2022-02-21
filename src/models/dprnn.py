import torch
import torch.nn as nn

from utils.model import choose_rnn
from utils.tasnet import choose_layer_norm

EPS = 1e-12

class DPRNN(nn.Module):
    def __init__(self, num_features, hidden_channels, num_blocks=6, norm=True, causal=False, rnn_type='lstm', eps=EPS):
        super().__init__()

        # Network confguration
        net = []

        for _ in range(num_blocks):
            net.append(DPRNNBlock(num_features, hidden_channels, norm=norm, causal=causal, rnn_type=rnn_type, eps=eps))

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

class DPRNNBlock(nn.Module):
    def __init__(self, num_features, hidden_channels, causal, norm=True, rnn_type='lstm', eps=EPS):
        super().__init__()

        self.intra_chunk_block = IntraChunkRNN(num_features, hidden_channels, norm=norm, rnn_type=rnn_type, eps=eps)
        self.inter_chunk_block = InterChunkRNN(num_features, hidden_channels, norm=norm, causal=causal, rnn_type=rnn_type, eps=eps)

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

class IntraChunkRNN(nn.Module):
    def __init__(self, num_features, hidden_channels, norm=True, rnn_type='lstm', eps=EPS):
        super().__init__()

        self.num_features, self.hidden_channels = num_features, hidden_channels
        num_directions = 2 # bi-direction
        self.norm = norm

        if rnn_type == 'lstm':
            self.rnn = choose_rnn(rnn_type, input_size=num_features, hidden_size=hidden_channels, batch_first=True, bidirectional=True)
        else:
            raise NotImplementedError("Not support {}.".format(rnn_type))

        self.fc = nn.Linear(num_directions*hidden_channels, num_features)

        if self.norm:
            norm_name = 'gLN'
            self.norm1d = choose_layer_norm(norm_name, num_features, causal=False, eps=eps)

    def forward(self, input):
        """
        Args:
            input (batch_size, num_features, S, chunk_size)
        Returns:
            output (batch_size, num_features, S, chunk_size)
        """
        num_features = self.num_features
        batch_size, _, S, chunk_size = input.size()

        self.rnn.flatten_parameters()

        residual = input # (batch_size, num_features, S, chunk_size)
        x = input.permute(0, 2, 3, 1).contiguous() # -> (batch_size, S, chunk_size, num_features)
        x = x.view(batch_size*S, chunk_size, num_features)
        x, _ = self.rnn(x) # (batch_size*S, chunk_size, num_features) -> (batch_size*S, chunk_size, num_directions*hidden_channels)
        x = self.fc(x) # -> (batch_size*S, chunk_size, num_features)
        x = x.view(batch_size, S*chunk_size, num_features) # (batch_size, S*chunk_size, num_features)
        x = x.permute(0, 2, 1).contiguous() # -> (batch_size, num_features, S*chunk_size)
        if self.norm:
            x = self.norm1d(x) # (batch_size, num_features, S*chunk_size)
        x = x.view(batch_size, num_features, S, chunk_size) # -> (batch_size, num_features, S, chunk_size)
        output = x + residual

        return output

class InterChunkRNN(nn.Module):
    def __init__(self, num_features, hidden_channels, causal, norm=True, rnn_type='lstm', eps=EPS):
        super().__init__()

        self.num_features, self.hidden_channels = num_features, hidden_channels
        self.norm = norm

        if rnn_type == 'lstm':
            self.rnn = choose_rnn(rnn_type, input_size=num_features, hidden_size=hidden_channels, batch_first=True, bidirectional=True)
        else:
            # Ensures LSTM
            raise NotImplementedError("Not support {}.".format(rnn_type))

        if causal: # uni-direction
            num_directions = 1
            self.rnn = choose_rnn(rnn_type, input_size=num_features, hidden_size=hidden_channels, batch_first=True, bidirectional=False)
        else: # bi-direction
            num_directions = 2
            self.rnn = choose_rnn(rnn_type, input_size=num_features, hidden_size=hidden_channels, batch_first=True, bidirectional=True)

        self.fc = nn.Linear(num_directions*hidden_channels, num_features)

        if self.norm:
            norm_name = 'cLN' if causal else 'gLN'
            self.norm1d = choose_layer_norm(norm_name, num_features, causal=causal, eps=eps)

    def forward(self, input):
        """
        Args:
            input (batch_size, num_features, S, chunk_size)
        Returns:
            output (batch_size, num_features, S, chunk_size)
        """
        num_features = self.num_features
        batch_size, _, S, chunk_size = input.size()

        self.rnn.flatten_parameters()

        residual = input # (batch_size, num_features, S, chunk_size)
        x = input.permute(0, 3, 2, 1).contiguous() # (batch_size, num_features, S, chunk_size) -> (batch_size, chunk_size, S, num_features)
        x = x.view(batch_size*chunk_size, S, num_features) # -> (batch_size*chunk_size, S, num_features)
        x, _ = self.rnn(x) # -> (batch_size*chunk_size, S, num_directions*hidden_channels)
        x = self.fc(x) # -> (batch_size*chunk_size, S, num_features)
        x = x.view(batch_size, chunk_size*S, num_features) # -> (batch_size, chunk_size*S, num_features)
        x = x.permute(0, 2, 1).contiguous() # -> (batch_size, num_features, chunk_size*S)
        if self.norm:
            x = self.norm1d(x) # -> (batch_size, num_features, chunk_size*S)
        x = x.view(batch_size, num_features, chunk_size, S) # -> (batch_size, num_features, chunk_size, S)
        x = x.permute(0, 1, 3, 2).contiguous() # -> (batch_size, num_features, S, chunk_size)

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
