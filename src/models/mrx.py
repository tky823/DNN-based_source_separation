import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.audio import build_window
from utils.model import choose_rnn
from models.umx import TransformBlock1d

__sources__ = ['music', 'speech', 'effects'] # ['bass', 'drums', 'other', 'vocals'] for MUSDB18
SAMPLE_RATE_MUSDB18 = 44100
EPS = 1e-12

class MultiResolutionCrossNet(nn.Module):
    """
    Reference "The Cocktail Fork Problem: Three-Stem Audio Separation for Real-World Soundtracks"
    """
    def __init__(self, in_channels, hidden_channels=512, num_layers=3, n_fft=None, hop_length=None, window_fn='hann', dropout=None, causal=False, rnn_type='lstm', sources=__sources__, eps=EPS):
        """
        Args:
            in_channels <int>: Input channels
            hidden_channels <int>: Hidden channels in LSTM
            num_layers <list<int>> or <int>: # of LSTM layers
            n_fft <list<int>>: FFT samples
            hop_length <int>: Hop length
            window_fn <str>: Window function
            dropout <list<float>> or <float>: Dropout rate in LSTM
            causal <bool>: Causality
            rnn_type <list<str>> or <str>: 'lstm'
            sources <list<str>>: Target sources
            eps <float>: Small value for numerical stability
        """
        super().__init__()

        if type(num_layers) is int:
            num_layers = [num_layers] * len(n_fft)

        if dropout is None:
            dropout = [None] * len(n_fft)
            dropout = [
                0.4 if _dropout is None and _num_layers > 1 else 0 for _num_layers, _dropout in zip(num_layers, dropout)
            ]

        if type(dropout) is float:
            dropout = [
                _dropout for _dropout in dropout
            ]
        else:
            dropout = [
                0.4 if _dropout is None and _num_layers > 1 else 0 for _num_layers, _dropout in zip(num_layers, dropout)
            ]

        if type(rnn_type) is str:
            rnn_type = [rnn_type] * len(n_fft)

        encoder_blocks, decoder_blocks = [], {}

        for _n_fft, _num_layers, _dropout, _rnn_type in zip(n_fft, num_layers, dropout, rnn_type):
            block = EncoderBlock(in_channels, hidden_channels, num_layers=_num_layers, dropout=_dropout, n_fft=_n_fft, hop_length=hop_length, window_fn=window_fn, causal=causal, rnn_type=_rnn_type, eps=eps)
            encoder_blocks.append(block)

        for source in sources:
            blocks = []
            for _n_fft in n_fft:
                block = DecoderBlock(2 * hidden_channels, in_channels, hidden_channels, _n_fft, hop_length=hop_length, window_fn=window_fn, eps=eps)
                blocks.append(block)
            decoder_blocks[source] = nn.ModuleList(blocks)

        self.encoder_blocks = nn.ModuleList(encoder_blocks)
        self.decoder_blocks = nn.ModuleDict(decoder_blocks)

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels

        self.n_fft, self.hop_length = n_fft, hop_length
        self.window_fn = window_fn

        self.num_layers = num_layers
        self.dropout = dropout
        self.causal = causal
        self.rnn_type = rnn_type

        self.sources = sources

        self.eps = eps

    def forward(self, input):
        in_channels = self.in_channels
        hidden_channels = self.hidden_channels
        T = input.size(-1)

        latent, x_ffts = [], []

        for idx, _n_fft in enumerate(self.n_fft):
            n_bins = _n_fft // 2 + 1
            x_latent = self.encoder_blocks[idx].stft(input)
            batch_size, _, _, _, n_frames = x_latent.size() # (batch_size, 1, in_channels, n_bins, n_frames)
            x_latent = x_latent.squeeze(dim=1)
            latent.append(x_latent)
            x = torch.abs(x_latent)
            x = x.permute(0, 3, 1, 2).contiguous() # (batch_size, n_frames, in_channels, n_bins)
            x = x.view(-1, in_channels * n_bins) # (batch_size * n_frames, in_channels * n_bins)
            x = self.encoder_blocks[idx].block(x)
            x = x.view(batch_size, n_frames, hidden_channels)
            x_ffts.append(x)

        x_fft_blocks = torch.stack(x_ffts, dim=0) # (len(n_ffts), batch_size, n_frames, hidden_channels)
        x_mean = x_fft_blocks.mean(dim=0) # (batch_size, n_frames, hidden_channels)

        x_ffts = []

        for idx, _n_fft in enumerate(self.n_fft):
            x_fft = x_fft_blocks[idx]
            x_rnn = self.encoder_blocks[idx].forward_rnn(x_mean) # (batch_size, n_frames, out_channels), where out_channels = hidden_channels
            x = torch.cat([x_fft, x_rnn], dim=2) # (batch_size, n_frames, hidden_channels + out_channels)
            x = x.view(batch_size * n_frames, 2 * hidden_channels)
            x_ffts.append(x)

        x_ffts = torch.stack(x_ffts, dim=0) # (len(n_ffts), batch_size * n_frames, hidden_channels + out_channels)
        x_ffts = x_ffts.mean(dim=0) # (batch_size * n_frames, hidden_channels + out_channels)
        output = []

        for source in self.sources:
            x_source = []

            for idx, _n_fft in enumerate(self.n_fft):
                n_bins = _n_fft // 2 + 1
                x_source_fft = self.decoder_blocks[source][idx].net(x_ffts) # (batch_size * n_frames, n_bins)
                x_source_fft = x_source_fft.view(batch_size, n_frames, in_channels, n_bins)
                x_source_fft = x_source_fft.permute(0, 2, 3, 1).contiguous() # (batch_size, in_channels, n_bins, n_frames)
                x_source_fft = self.decoder_blocks[source][idx].transform_affine(x_source_fft)
                mask = self.decoder_blocks[source][idx].relu2d(x_source_fft)
                x_source_fft = mask * latent[idx]
                x_source_fft = self.decoder_blocks[source][idx].istft(x_source_fft, length=T)
                x_source.append(x_source_fft)

            x_source = torch.stack(x_source, dim=0)
            x_source = x_source.sum(dim=0)
            output.append(x_source)

        output = torch.stack(output, dim=1)

        return output

    def get_config(self):
        config = {
            'in_channels': self.in_channels,
            'hidden_channels': self.hidden_channels,
            'num_layers': self.num_layers,
            'n_fft': self.n_fft, 'hop_length': self.hop_length,
            'window_fn': self.window_fn,
            'dropout': self.dropout,
            'causal': self.causal,
            'rnn_type': self.rnn_type,
            'sources': self.sources,
            'eps': self.eps
        }

        return config

    @classmethod
    def build_from_config(cls, config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        in_channels = config['in_channels']

        hidden_channels = config['hidden_channels']
        num_layers = config['num_layers']

        n_fft, hop_length = config['n_fft'], config['hop_length']
        window_fn = config['window_fn']

        dropout = config['dropout']
        causal = config['causal']
        rnn_type = config['rnn_type']

        sources = config['sources']

        eps = config.get('eps') or EPS

        model = cls(
            in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            n_fft=n_fft, hop_length=hop_length, window_fn=window_fn,
            dropout=dropout,
            causal=causal,
            rnn_type=rnn_type,
            sources=sources,
            eps=eps
        )

        return model

    @classmethod
    def build_model(cls, model_path, load_state_dict=False):
        config = torch.load(model_path, map_location=lambda storage, loc: storage)

        in_channels = config['in_channels']
        hidden_channels = config['hidden_channels']
        num_layers = config['num_layers']

        n_fft, hop_length = config['n_fft'], config['hop_length']
        window_fn = config['window_fn']

        dropout = config['dropout']
        causal = config['causal']
        rnn_type = config['rnn_type']

        sources = config['sources']

        eps = config.get('eps') or EPS

        model = cls(
            in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            n_fft=n_fft, hop_length=hop_length, window_fn=window_fn,
            dropout=dropout,
            causal=causal,
            rnn_type=rnn_type,
            sources=sources,
            eps=eps
        )

        if load_state_dict:
            model.load_state_dict(config['state_dict'])

        return model

    @property
    def num_parameters(self):
        _num_parameters = 0

        for p in self.parameters():
            if p.requires_grad:
                _num_parameters += p.numel()

        return _num_parameters

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels=512, num_layers=3, n_fft=None, hop_length=None, window_fn='hann', dropout=None, causal=False, rnn_type='lstm', eps=EPS):
        super().__init__()

        n_bins = n_fft // 2 + 1

        self.stft = STFT(n_fft, hop_length=hop_length, window_fn=window_fn)
        self.block = TransformBlock1d(in_channels * n_bins, hidden_channels, bias=False, nonlinear='tanh')

        if causal:
            bidirectional = False
            rnn_hidden_channels = hidden_channels
        else:
            assert hidden_channels % 2 == 0, "hidden_channels is expected even number, but given {}.".format(hidden_channels)

            bidirectional = True
            rnn_hidden_channels = hidden_channels // 2

        self.rnn = choose_rnn(rnn_type, input_size=hidden_channels, hidden_size=rnn_hidden_channels, num_layers=num_layers, bidirectional=bidirectional, batch_first=True, dropout=dropout)

        self.scale_in, self.bias_in = nn.Parameter(torch.Tensor(n_bins,), requires_grad=True), nn.Parameter(torch.Tensor(n_bins,), requires_grad=True)

        self.eps = eps

        self._reset_parameters()

    def _reset_parameters(self):
        self.scale_in.data.fill_(1)
        self.bias_in.data.zero_()

    def forward(self, input):
        raise NotImplementedError

    def transform_affine(self, input):
        """
        Args:
            input: (batch_size, n_channels, n_bins, n_frames)
        Returns:
            output: (batch_size, n_channels, n_bins, n_frames)
        """
        eps = self.eps

        output = (input - self.bias_in.unsqueeze(dim=1)) / (torch.abs(self.scale_in.unsqueeze(dim=1)) + eps) # (batch_size, n_channels, n_bins, n_frames)

        return output

    def forward_rnn(self, input):
        """
        Args:
            input: (batch_size, n_frames, in_channels)
        Returns:
            output: (batch_size, n_frames, out_channels)
        """
        self.rnn.flatten_parameters()

        output, _ = self.rnn(input) # (batch_size, n_frames, out_channels)

        return output

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, n_fft=None, hop_length=None, window_fn='hann', nonlinear='relu', eps=EPS):
        super().__init__()

        n_bins = n_fft // 2 + 1

        net = []
        net.append(TransformBlock1d(in_channels, hidden_channels, bias=False, nonlinear=nonlinear))
        net.append(TransformBlock1d(hidden_channels, out_channels * n_bins, bias=False))

        self.net = nn.Sequential(*net)
        self.relu2d = nn.ReLU()
        self.istft = iSTFT(n_fft, hop_length=hop_length, window_fn=window_fn)

        self.scale_out, self.bias_out = nn.Parameter(torch.Tensor(n_bins,), requires_grad=True), nn.Parameter(torch.Tensor(n_bins,), requires_grad=True)

        self._reset_parameters()

    def _reset_parameters(self):
        self.scale_out.data.fill_(1)
        self.bias_out.data.zero_()

    def forward(self, input):
        raise NotImplementedError

    def transform_affine(self, input):
        """
        Args:
            input: (batch_size, n_channels, n_bins, n_frames)
        Returns:
            output: (batch_size, n_channels, n_bins, n_frames)
        """
        output = self.scale_out.unsqueeze(dim=1) * input + self.bias_out.unsqueeze(dim=1)

        return output

class STFT(nn.Module):
    def __init__(self, n_fft, hop_length=None, window_fn='hann'):
        super().__init__()

        self.n_fft, self.hop_length = n_fft, hop_length

        if hop_length == n_fft:
            window = torch.ones(n_fft)
        else:
            window = build_window(n_fft, window_fn=window_fn)

        self.window = nn.Parameter(window, requires_grad=False)

    def forward(self, input):
        """
        Args:
            input: (batch_size, *, T)
        Returns:
            output: (batch_size, *, n_bins, n_frames)
        """
        n_fft, hop_length = self.n_fft, self.hop_length

        channels = input.size()[:-1]

        input = input.view(-1, input.size(-1))
        input = F.pad(input, (n_fft // 2, n_fft // 2 + hop_length))
        x = torch.stft(input, n_fft=n_fft, hop_length=hop_length, window=self.window, center=False, onesided=True, return_complex=True)
        output = x.view(*channels, *x.size()[-2:])

        return output

class iSTFT(nn.Module):
    def __init__(self, n_fft, hop_length=None, window_fn='hann'):
        super().__init__()

        self.n_fft, self.hop_length = n_fft, hop_length

        if hop_length == n_fft:
            window = torch.ones(n_fft)
        else:
            window = build_window(n_fft, window_fn=window_fn)

        self.window = nn.Parameter(window, requires_grad=False)

    def forward(self, input, length=None):
        """
        Args:
            input: (batch_size, *, n_bins, n_frames)
        Returns:
            output: (batch_size, *, T)
        """
        n_fft, hop_length = self.n_fft, self.hop_length

        channels = input.size()[:-2]

        input = input.view(-1, *input.size()[-2:])
        x = torch.istft(input, n_fft=n_fft, hop_length=hop_length, window=self.window, center=True, onesided=True, return_complex=False)
        output = x.view(*channels, -1)

        if length is not None:
            output = output[..., :length]

        return output

def _test_mrx():
    batch_size = 6
    in_channels = 2
    T = 1025
    n_fft, hop_length = [32, 64, 128], 32

    model = MultiResolutionCrossNet(in_channels, n_fft=n_fft, hop_length=hop_length)

    input = torch.randn(batch_size, 1, in_channels, T)
    output = model(input)

    print(model)
    print(input.size(), output.size())

if __name__ == '__main__':
    torch.manual_seed(111)

    _test_mrx()