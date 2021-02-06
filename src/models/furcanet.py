import torch
import torch.nn as nn

from models.glu import GLU1d

EPS=1e-12

class FurcaNetBase(nn.Module):
    def __init__(self, num_conv_blocks, num_lstm_blocks, causal=False):
        super().__init__()
        
        self.num_conv_blocks, self.num_lstm_blocks = num_conv_blocks, num_lstm_blocks
        self.causal = causal
        num_directions = 2 # bi-direction
        
        self.gcn = GatedConvNet()
        self.stacked_lstm = nn.LSTM(num_features, hidden_channels//num_directions, layers=num_lstm_blocks, batch_first=True, bidirectional=True)
        self.dnn = nn.Linear()


class GatedConvNet(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=None, num_layers=10, causal=False, nonlinear=None, norm=True, eps=EPS):
        super().__init__()
        
        self.num_blocks = num_blocks
        
        net = []
        
        for idx in range(num_blocks):
            net.append(GLU1d(in_channels, out_channels))
        
        self.net = nn.Sequential(*net)
        
    def forward(self, input):
        num_blocks = self.num_blocks
        
        x = input
        skip_connection = 0
        
        for idx in range(num_blocks):
            x, skip = self.net[idx](x)
            skip_connection = skip_connection + skip

        output = skip_connection
        
        return output

