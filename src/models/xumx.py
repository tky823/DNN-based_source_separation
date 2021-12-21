import os

import yaml
import torch
import torch.nn as nn

from utils.audio import build_window
from algorithm.frequency_mask import multichannel_wiener_filter
from transforms.stft import stft, istft
from models.umx import OpenUnmix

__sources__ = ['bass', 'drums', 'other', 'vocals']
SAMPLE_RATE_MUSDB18 = 44100
EPS = 1e-12

"""
CrossNet-Open-Unmix
    Reference: "All for One and One for All: Improving Music Separation by Bridging Networks"
    See https://arxiv.org/abs/2010.04228
"""
class CrossNetOpenUnmix(nn.Module):
    pretrained_model_ids = {
        "musdb18": {
            SAMPLE_RATE_MUSDB18: {
                "paper": "1yQC00DFvHgs4U012Wzcg69lvRxw5K9Jj"
            }
        },
        "musdb18hq": {
            SAMPLE_RATE_MUSDB18: {
                "paper": None
            }
        }
    }
    def __init__(self, in_channels, hidden_channels=512, num_layers=3, n_bins=None, max_bin=None, dropout=None, causal=False, rnn_type='lstm', bridge=True, sources=__sources__, eps=EPS):
        """
        Args:
            in_channels <int>: Input channels
            hidden_channels <int>: Hidden channels in LSTM
            num_layers <int>: # of LSTM layers
            n_bins <int>: # of frequency bins
            max_bin <int>: If none, max_bin = n_bins
            dropout <float>: Dropout rate in LSTM
            causal <bool>: Causality
            rnn_type <str>: 'lstm'
            bridge <bool>: Bridging network.
            sources <list<str>>: Target sources
            eps <float>: Small value for numerical stability
        """
        super().__init__()

        net = {}

        for source in sources:
            net[source] = OpenUnmix(in_channels, hidden_channels, num_layers=num_layers, n_bins=n_bins, max_bin=max_bin, dropout=dropout, causal=causal, rnn_type=rnn_type, eps=eps)

        self.backbone = nn.ModuleDict(net)

        # Hyperparameters
        self.in_channels, self.n_bins = in_channels, n_bins
        self.hidden_channels, self.out_channels = hidden_channels, hidden_channels
        self.num_layers = num_layers
        self.max_bin = max_bin

        self.dropout = dropout
        self.causal = causal
        self.rnn_type = rnn_type
        self.bridge = bridge

        self.sources = sources

        self.eps = eps
        
    def forward(self, input):
        """
        Args:
            input <torch.Tensor>: (batch_size, 1, in_channels, n_bins, n_frames)
        Returns:
            output <torch.Tensor>: (batch_size, n_sources, in_channels, n_bins, n_frames)
        """
        n_bins, max_bin = self.n_bins, self.max_bin

        input = input.squeeze(dim=1)

        if max_bin == n_bins:
            x_valid = input
        else:
            sections = [max_bin, n_bins - max_bin]
            x_valid, _ = torch.split(input, sections, dim=2)

        for source in self.sources:
            self.backbone[source].rnn.flatten_parameters()

        if self.bridge:
            output = self.forward_bridge(input, x_valid)
        else:
            output = self.forward_no_bridge(input, x_valid)

        return output

    def forward_no_bridge(self, input, x_valid):
        n_bins, max_bin = self.n_bins, self.max_bin
        in_channels, hidden_channels, out_channels = self.in_channels, self.hidden_channels, self.out_channels

        batch_size, _, _, n_frames = x_valid.size()

        x_sources = []

        for source in self.sources:
            x_source = self.backbone[source].transform_affine_in(x_valid) # (batch_size, n_channels, max_bin, n_frames)
            x_source = x_source.permute(0, 3, 1, 2).contiguous() # (batch_size, n_frames, n_channels, max_bin)
            x_source = x_source.view(batch_size * n_frames, in_channels * max_bin)
            x_source = self.backbone[source].block(x_source) # (batch_size * n_frames, hidden_channels)
            x_source = x_source.view(batch_size, n_frames, hidden_channels)
            x_sources.append(x_source)

        x_sources_block = torch.stack(x_sources, dim=0) # (n_sources, batch_size, n_frames, hidden_channels)
        x_sources = []

        for idx, source in enumerate(self.sources):
            x_source = x_sources_block[idx]
            x_source_lstm, _ = self.backbone[source].rnn(x_source) # (batch_size, n_frames, out_channels)
            x_source = torch.cat([x_source, x_source_lstm], dim=2) # (batch_size, n_frames, hidden_channels + out_channels)
            x_source = x_source.view(batch_size * n_frames, hidden_channels + out_channels)
            x_sources.append(x_source)

        x_sources = torch.stack(x_sources, dim=0) # (n_sources, batch_size * n_frames, hidden_channels + out_channels)
        output = []

        for idx, source in enumerate(self.sources):
            x_source = x_sources[idx]
            x_source_full = self.backbone[source].net(x_source) # (batch_size * n_frames, n_bins)
            x_source_full = x_source_full.view(batch_size, n_frames, in_channels, n_bins)
            x_source_full = x_source_full.permute(0, 2, 3, 1).contiguous() # (batch_size, in_channels, n_bins, n_frames)
            x_source_full = self.backbone[source].transform_affine_out(x_source_full) # (batch_size, n_channels, max_bin, n_frames)
            x_source_full = self.backbone[source].relu2d(x_source_full)
            x_source = x_source_full * input
            output.append(x_source)

        output = torch.stack(output, dim=1) # (batch_size, n_sources, in_channels, n_bins, n_frames)

        return output

    def forward_bridge(self, input, x_valid):
        n_bins, max_bin = self.n_bins, self.max_bin
        in_channels, hidden_channels, out_channels = self.in_channels, self.hidden_channels, self.out_channels

        batch_size, _, _, n_frames = x_valid.size()

        x_sources = []

        for source in self.sources:
            x_source = self.backbone[source].transform_affine_in(x_valid) # (batch_size, n_channels, max_bin, n_frames)
            x_source = x_source.permute(0, 3, 1, 2).contiguous() # (batch_size, n_frames, n_channels, max_bin)
            x_source = x_source.view(batch_size * n_frames, in_channels * max_bin)
            x_source = self.backbone[source].block(x_source) # (batch_size * n_frames, hidden_channels)
            x_source = x_source.view(batch_size, n_frames, hidden_channels)
            x_sources.append(x_source)

        x_sources_block = torch.stack(x_sources, dim=0) # (n_sources, batch_size, n_frames, hidden_channels)
        x_mean = x_sources_block.mean(dim=0) # (batch_size, n_frames, hidden_channels)
        x_sources = []

        for idx, source in enumerate(self.sources):
            x_source = x_sources_block[idx]
            x_source_rnn, _ = self.backbone[source].rnn(x_mean) # (batch_size, n_frames, out_channels)
            x_source = torch.cat([x_source, x_source_rnn], dim=2) # (batch_size, n_frames, hidden_channels + out_channels)
            x_source = x_source.view(batch_size * n_frames, hidden_channels + out_channels)
            x_sources.append(x_source)

        x_sources = torch.stack(x_sources, dim=0) # (n_sources, batch_size * n_frames, hidden_channels + out_channels)
        x = x_sources.mean(dim=0) # (batch_size * n_frames, hidden_channels + out_channels)
        output = []

        for source in self.sources:
            x_source_full = self.backbone[source].net(x) # (batch_size * n_frames, n_bins)
            x_source_full = x_source_full.view(batch_size, n_frames, in_channels, n_bins)
            x_source_full = x_source_full.permute(0, 2, 3, 1).contiguous() # (batch_size, in_channels, n_bins, n_frames)
            x_source_full = self.backbone[source].transform_affine_out(x_source_full) # (batch_size, in_channels, n_bins, n_frames)
            x_source_full = self.backbone[source].relu2d(x_source_full)
            x_source = x_source_full * input
            output.append(x_source)

        output = torch.stack(output, dim=1) # (batch_size, n_sources, in_channels, n_bins, n_frames)

        return output

    def get_config(self):
        config = {
            'in_channels': self.in_channels,
            'hidden_channels': self.hidden_channels,
            'num_layers': self.num_layers,
            'n_bins': self.n_bins,
            'max_bin': self.max_bin,
            'dropout': self.dropout,
            'causal': self.causal,
            'rnn_type': self.rnn_type,
            'bridge': self.bridge,
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
        n_bins, max_bin = config['n_bins'], config['max_bin']
        dropout = config['dropout']
        causal = config['causal']
        rnn_type = config['rnn_type']
        bridge = config['bridge']

        sources = config['sources']

        eps = config.get('eps') or EPS

        model = cls(
            in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            n_bins=n_bins, max_bin=max_bin,
            dropout=dropout,
            causal=causal,
            sources=sources,
            rnn_type=rnn_type,
            bridge=bridge,
            eps=eps
        )

        return model

    @classmethod
    def build_model(cls, model_path, load_state_dict=False):
        config = torch.load(model_path, map_location=lambda storage, loc: storage)

        in_channels = config['in_channels']
        hidden_channels = config['hidden_channels']
        num_layers = config['num_layers']
        n_bins, max_bin = config['n_bins'], config['max_bin']
        dropout = config['dropout']
        causal = config['causal']
        rnn_type = config['rnn_type']
        bridge = config['bridge']

        sources = config['sources']

        eps = config.get('eps') or EPS

        model = cls(
            in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            n_bins=n_bins, max_bin=max_bin,
            dropout=dropout,
            causal=causal,
            rnn_type=rnn_type,
            bridge=bridge,
            sources=sources,
            eps=eps
        )

        if load_state_dict:
            model.load_state_dict(config['state_dict'])

        return model

    @classmethod
    def build_from_pretrained(cls, root="./pretrained", quiet=False, load_state_dict=True, **kwargs):
        from utils.utils import download_pretrained_model_from_google_drive

        task = kwargs.get('task')

        if not task in cls.pretrained_model_ids:
            raise KeyError("Invalid task ({}) is specified.".format(task))

        pretrained_model_ids_task = cls.pretrained_model_ids[task]
        additional_attributes = {}

        if task in ['musdb18', 'musdb18hq']:
            sample_rate = kwargs.get('sample_rate') or SAMPLE_RATE_MUSDB18
            config = kwargs.get('config') or "paper"
            model_choice = kwargs.get('model_choice') or 'best'

            model_id = pretrained_model_ids_task[sample_rate][config]
            download_dir = os.path.join(root, cls.__name__, task, "sr{}".format(sample_rate), config)
        else:
            raise NotImplementedError("Not support task={}.".format(task))

        additional_attributes.update({
            'sample_rate': sample_rate
        })

        model_path = os.path.join(download_dir, "model", "{}.pth".format(model_choice))

        if not os.path.exists(model_path):
            download_pretrained_model_from_google_drive(model_id, download_dir, quiet=quiet)

        config = torch.load(model_path, map_location=lambda storage, loc: storage)
        model = cls.build_model(model_path, load_state_dict=load_state_dict)

        if task in ['musdb18']:
            additional_attributes.update({
                'sources': config['sources'],
                'n_sources': len(config['sources']),
                'n_fft': config['n_fft'], 'hop_length': config['hop_length'],
                'window_fn': config['window_fn'],
            })

        for key, value in additional_attributes.items():
            setattr(model, key, value)

        return model

    @classmethod
    def TimeDomainWrapper(cls, base_model, n_fft, hop_length=None, window_fn='hann', eps=EPS):
        return CrossNetOpenUnmixTimeDomainWrapper(base_model, n_fft, hop_length=hop_length, window_fn=window_fn, eps=eps)

    @property
    def num_parameters(self):
        _num_parameters = 0

        for p in self.parameters():
            if p.requires_grad:
                _num_parameters += p.numel()

        return _num_parameters

class CrossNetOpenUnmixTimeDomainWrapper(nn.Module):
    def __init__(self, base_model: CrossNetOpenUnmix, n_fft, hop_length=None, window_fn='hann', eps=EPS):
        super().__init__()

        self.base_model = base_model

        if hop_length is None:
            hop_length = n_fft // 4

        self.n_fft, self.hop_length = n_fft, hop_length
        window = build_window(n_fft, window_fn=window_fn)
        self.window = nn.Parameter(window, requires_grad=False)

        self.sources = self.base_model.sources
        self.eps = eps

    def forward(self, input, iteration=1):
        """
        Args:
            input <torch.Tensor>: (batch_size, 1, in_channels, T)
            iteration <int>: Iteration of EM algorithm
        Returns:
            output <torch.Tensor>: (batch_size, n_sources, in_channels, T)
        """
        assert input.dim() == 4, "input is expected 4D input."

        T = input.size(-1)
        eps = self.eps

        mixture_spectrogram = stft(input, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window, onesided=True, return_complex=True)
        mixture_amplitude = torch.abs(mixture_spectrogram)

        estimated_amplitude = self.base_model(mixture_amplitude)
        estimated_spectrogram = multichannel_wiener_filter(mixture_spectrogram, estimated_sources_amplitude=estimated_amplitude, iteration=iteration, eps=eps)
        output = istft(estimated_spectrogram, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window, onesided=True, return_complex=False, length=T)

        return output

def _test_crossnet_openunmix():
    batch_size = 6
    in_channels = 2
    n_bins, max_bin = 2049, 1487
    n_frames = 100
    dropout = 0.4

    input = torch.randn(batch_size, 1, in_channels, n_bins, n_frames)

    print('-'*10, "Non causal", '-'*10)
    causal = False
    model = CrossNetOpenUnmix(in_channels=in_channels, n_bins=n_bins, max_bin=max_bin, dropout=dropout, causal=causal)
    output = model(input)

    print(model)
    print(model.num_parameters)
    print(input.size(), output.size())

if __name__ == '__main__':
    torch.manual_seed(111)

    print("="*10, "Cross-Net Open-Unmix (X-UMX)", "="*10)
    _test_crossnet_openunmix()