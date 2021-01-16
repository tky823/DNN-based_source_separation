import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils_tasnet import choose_bases
from models.dprnn_tasnet import Segment1d, OverlapAdd1d
from models.mulcat_rnn import MulCatDPRNN

EPS=1e-12

"""
"Voice Separation with an Unknown Number of Multiple Speakers"
https://arxiv.org/abs/2003.01531

We have to name the model.
"""

class Separator(nn.Module):
    def __init__(self, num_features, hidden_channels=128, chunk_size=100, hop_size=50, num_blocks=6, causal=True, n_sources=2, eps=EPS):
        super().__init__()
        
        self.num_features, self.n_sources = num_features, n_sources
        self.chunk_size, self.hop_size = chunk_size, hop_size
        self.num_blocks = num_blocks
        
        self.segment1d = Segment1d(chunk_size, hop_size)
        self.mulcat_dprnn = MulCatDPRNN(num_features, hidden_channels, num_blocks=num_blocks, causal=causal, n_sources=n_sources, eps=eps)
        self.overlap_add1d = OverlapAdd1d(chunk_size, hop_size)

    def forward(self, input):
        """
        Args:
            input (batch_size, num_features, n_frames)
        Returns:
            output (batch_size, n_sources, num_features, n_frames)
        """
        num_features, n_sources = self.num_features, self.n_sources
        chunk_size, hop_size = self.chunk_size, self.hop_size
        num_blocks_half = self.num_blocks//2
        batch_size, num_features, n_frames = input.size()
        
        padding = (hop_size-(n_frames-chunk_size)%hop_size)%hop_size
        padding_left = padding//2
        padding_right = padding - padding_left
        
        x = F.pad(input, (padding_left, padding_right))
        x = self.segment1d(x)
        x = self.mulcat_dprnn(x) # (batch_size, num_blocks//2, n_sources, num_features, S, chunk_size)
        x = x.view(batch_size*num_blocks_half*n_sources, num_features, -1, chunk_size)
        x = self.overlap_add1d(x)
        x = F.pad(x, (-padding_left, -padding_right))
        output = x.view(batch_size, num_blocks_half, n_sources, num_features, n_frames)
        
        return output


if __name__ == '__main__':
    batch_size = 4

    num_features, hidden_channels = 8, 32
    n_frames = 20
    chunk_size, hop_size = 4, 2
    num_blocks = 6
    causal = False
    n_sources = 2

    input = torch.randint(0, 10, (batch_size, num_features, n_frames), dtype=torch.float)

    model = Separator(num_features, hidden_channels, chunk_size=chunk_size, hop_size=hop_size, num_blocks=num_blocks, causal=causal, n_sources=n_sources)
    print(model)

    output = model(input)
    print(input.size(), output.size())
    print()