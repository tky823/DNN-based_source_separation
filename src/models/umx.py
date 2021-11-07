import yaml
import torch
import torch.nn as nn

from utils.model import choose_nonlinear, choose_rnn

__sources__ = ['bass', 'drums', 'other', 'vocals']
SAMPLE_RATE_MUSDB18 = 44100
EPS = 1e-12
__pretrained_model_ids__ = {
    "musdb18": {
        SAMPLE_RATE_MUSDB18: {
            "paper": "1sqlK26fLJ6ns-NOxCrxhwI92wv45QPCB"
        }
    },
    "musdb18hq": {
        SAMPLE_RATE_MUSDB18: {
            "paper": "18pj2ubYnZPSQWPpHaREAcbmrNzEihNHO"
        }
    }
}

"""
Reference: https://github.com/sigsep/open-unmix-pytorch
"""

class ParallelOpenUnmix(nn.Module):
    def __init__(self, modules):
        super().__init__()

        if isinstance(modules, nn.ModuleDict):
            pass
        elif isinstance(modules, dict):
            modules = nn.ModuleDict(modules)
        else:
            raise TypeError("Type of `modules` is expected nn.ModuleDict or dict, but given {}.".format(type(modules)))
    
        in_channels = None

        for key in modules.keys():
            module = modules[key]
            if not isinstance(module, OpenUnmix):
                raise ValueError("All modules must be OpenUnmix.")
            
            if in_channels is None:
                in_channels = module.in_channels
            else:
                assert in_channels == module.in_channels, "`in_channels` are different among modules."
        
        self.net = modules

        self.in_channels = in_channels

    def forward(self, input, target=None):
        if type(target) is not str:
            raise TypeError("`target` is expected str, but given {}".format(type(target)))
        
        output = self.net[target](input)

        return output
    
    @property
    def num_parameters(self):
        _num_parameters = 0
        
        for p in self.parameters():
            if p.requires_grad:
                _num_parameters += p.numel()
                
        return _num_parameters

"""
Open-Unmix
    Reference: "Open-unmix: a reference implementation for source separation"
    See https://hal.inria.fr/hal-02293689/document
"""
class OpenUnmix(nn.Module):
    def __init__(self, in_channels, hidden_channels=512, num_layers=3, n_bins=None, max_bin=None, dropout=None, causal=False, rnn_type='lstm', eps=EPS):
        """
        Args:
            in_channels <int>: Input channels
            hidden_channels <int>: Hidden channels in LSTM
            num_layers <int>: # of LSTM layers
            n_bins <int>: # of frequency bins
            max_bin <int>: If none, max_bin = n_bins
            dropout <float>: Dropout rate in LSTM
            causal <bool>: Causality
            eps <float>: Small value for numerical stability
        """
        super().__init__()

        if n_bins is None:
            raise ValueError("Specify `n_bins`.")

        if max_bin is None:
            max_bin = n_bins
        
        if dropout is None:
            dropout = 0.4 if num_layers > 1 else 0

        self.block = TransformBlock1d(in_channels * max_bin, hidden_channels, bias=False, nonlinear='tanh')

        rnn_in_channels = hidden_channels

        if causal:
            bidirectional = False
            rnn_hidden_channels = hidden_channels
            out_channels = hidden_channels
        else:
            assert hidden_channels % 2 == 0, "hidden_channels is expected even number, but given {}.".format(hidden_channels)

            bidirectional = True
            rnn_hidden_channels = hidden_channels // 2
            out_channels = hidden_channels

        self.rnn = choose_rnn(rnn_type, input_size=rnn_in_channels, hidden_size=rnn_hidden_channels, num_layers=num_layers, bidirectional=bidirectional, batch_first=True, dropout=dropout)

        net = []
        net.append(TransformBlock1d(hidden_channels + out_channels, hidden_channels, bias=False, nonlinear='relu'))
        net.append(TransformBlock1d(hidden_channels, in_channels * n_bins, bias=False))

        self.net = nn.Sequential(*net)
        self.relu2d = nn.ReLU()

        self.scale_in, self.bias_in = nn.Parameter(torch.Tensor(max_bin,), requires_grad=True), nn.Parameter(torch.Tensor(max_bin,), requires_grad=True)
        self.scale_out, self.bias_out = nn.Parameter(torch.Tensor(n_bins,), requires_grad=True), nn.Parameter(torch.Tensor(n_bins,), requires_grad=True)

        # Hyperparameters
        self.in_channels, self.n_bins = in_channels, n_bins
        self.hidden_channels, self.out_channels = hidden_channels, out_channels
        self.num_layers = num_layers
        self.max_bin = max_bin

        self.dropout = dropout
        self.causal = causal
        self.rnn_type = rnn_type
        
        self.eps = eps

        self._reset_parameters()
    
    def _reset_parameters(self):
        self.scale_in.data.fill_(1)
        self.bias_in.data.zero_()
        self.scale_out.data.fill_(1)
        self.bias_out.data.zero_()

    def forward(self, input):
        """
        Args:
            input: (batch_size, in_channels, n_bins, n_frames)
        Returns:
            output: (batch_size, in_channels, n_bins, n_frames)
        """
        n_bins, max_bin = self.n_bins, self.max_bin
        in_channels, hidden_channels, out_channels = self.in_channels, self.hidden_channels, self.out_channels

        batch_size, _, _, n_frames = input.size()

        self.rnn.flatten_parameters()

        if max_bin == n_bins:
            x_valid = input
        else:
            sections = [max_bin, n_bins - max_bin]
            x_valid, _ = torch.split(input, sections, dim=2)

        x = self.transform_affine_in(x_valid) # (batch_size, n_channels, max_bin, n_frames)
        x = x.permute(0, 3, 1, 2).contiguous() # (batch_size, n_frames, n_channels, max_bin)
        x = x.view(batch_size * n_frames, in_channels * max_bin)
        x = self.block(x) # (batch_size * n_frames, hidden_channels)
        x = x.view(batch_size, n_frames, hidden_channels)
        x_rnn, _ = self.rnn(x) # (batch_size, n_frames, out_channels)
        x = torch.cat([x, x_rnn], dim=2) # (batch_size, n_frames, hidden_channels + out_channels)
        x = x.view(batch_size * n_frames, hidden_channels + out_channels)
        x_full = self.net(x) # (batch_size * n_frames, n_bins)
        x_full = x_full.view(batch_size, n_frames, in_channels, n_bins)
        x_full = x_full.permute(0, 2, 3, 1).contiguous() # (batch_size, in_channels, n_bins, n_frames)

        x_full = self.transform_affine_out(x_full)
        x_full = self.relu2d(x_full)

        output = x_full * input

        return output
    
    def transform_affine_in(self, input):
        """
        Args:
            input: (batch_size, n_channels, max_bin, n_frames)
        Rreturns:
            output: (batch_size, n_channels, max_bin, n_frames)
        """
        eps = self.eps

        output = (input - self.bias_in.unsqueeze(dim=1)) / (torch.abs(self.scale_in.unsqueeze(dim=1)) + eps) # (batch_size, n_channels, max_bin, n_frames)

        return output

    def transform_affine_out(self, input):
        """
        Args:
            input: (batch_size, n_channels, n_bins, n_frames)
        Rreturns:
            output: (batch_size, n_channels, n_bins, n_frames)
        """
        output = self.scale_out.unsqueeze(dim=1) * input + self.bias_out.unsqueeze(dim=1)

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
        rnn_type = config.get('rnn_type') or 'lstm'

        eps = config.get('eps') or EPS

        model = cls(
            in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            n_bins=n_bins, max_bin=max_bin,
            dropout=dropout,
            causal=causal,
            rnn_type=rnn_type,
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
        rnn_type = config.get('rnn_type') or 'lstm'

        eps = config.get('eps') or EPS
        
        model = cls(
            in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            n_bins=n_bins, max_bin=max_bin,
            dropout=dropout,
            causal=causal,
            rnn_type=rnn_type,
            eps=eps
        )
        
        if load_state_dict:
            model.load_state_dict(config['state_dict'])
        
        return model
    
    @classmethod
    def build_from_pretrained(cls, root="./pretrained", target='vocals', quiet=False, load_state_dict=True, **kwargs):
        import os
        
        from utils.utils import download_pretrained_model_from_google_drive

        task = kwargs.get('task')

        if not task in __pretrained_model_ids__:
            raise KeyError("Invalid task ({}) is specified.".format(task))
            
        pretrained_model_ids_task = __pretrained_model_ids__[task]
        
        if task in ['musdb18', 'musdb18hq']:
            sample_rate = kwargs.get('sr') or kwargs.get('sample_rate') or SAMPLE_RATE_MUSDB18
            config = kwargs.get('config') or "paper"
            model_choice = kwargs.get('model_choice') or 'best'

            model_id = pretrained_model_ids_task[sample_rate][config]
            download_dir = os.path.join(root, cls.__name__, task, "sr{}".format(sample_rate), config)
        else:
            raise NotImplementedError("Not support task={}.".format(task))
        
        model_path = os.path.join(download_dir, "model", target, "{}.pth".format(model_choice))

        if not os.path.exists(model_path):
            download_pretrained_model_from_google_drive(model_id, download_dir, quiet=quiet)
        
        model = cls.build_model(model_path, load_state_dict=load_state_dict)

        return model
    
    @property
    def num_parameters(self):
        _num_parameters = 0
        
        for p in self.parameters():
            if p.requires_grad:
                _num_parameters += p.numel()
        
        return _num_parameters

class TransformBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, nonlinear=None, eps=EPS):
        super().__init__()
        self.fc = nn.Linear(in_channels, out_channels, bias=bias)
        self.norm1d = nn.BatchNorm1d(out_channels, eps=eps)

        if nonlinear is None:
            self.nonlinear = False
        else:
            self.nonlinear = True
            self.nonlinear1d = choose_nonlinear(nonlinear)
    
    def forward(self, input):
        """
        Args:
            input: (batch_size, in_channels)
        Returns:
            output: (batch_size, out_channels)
        """
        x = self.fc(input)
        x = self.norm1d(x)

        if self.nonlinear:
            output = self.nonlinear1d(x)
        else:
            output = x

        return output

def _test_openunmix():
    batch_size = 4
    in_channels = 2
    n_bins, max_bin = 2049, 1487
    n_frames = 100
    dropout = 0.4

    input = torch.randn(batch_size, in_channels, n_bins, n_frames)

    print('-'*10, "Non causal", '-'*10)
    causal = False
    model = OpenUnmix(in_channels=in_channels, n_bins=n_bins, max_bin=max_bin, dropout=dropout, causal=causal)
    output = model(input)

    print(model)
    print(input.size(), output.size())
    print()

    print('-'*10, "Causal", '-'*10)
    causal = True
    model = OpenUnmix(in_channels=in_channels, n_bins=n_bins, max_bin=max_bin, dropout=dropout, causal=causal)
    output = model(input)

    print(model)
    print(input.size(), output.size())

if __name__ == '__main__':
    torch.manual_seed(111)

    print("="*10, "Open-Unmix (UMX)", "="*10)
    _test_openunmix()