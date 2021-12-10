import os

import torch
import torch.nn as nn

from utils.model import choose_rnn

EPS = 1e-12

class DeepEmbedding(nn.Module):
    pretrained_model_ids = {
        "wsj0-mix": {
            8000: {
                2: "1-34LWrdBKrZpREwHAJfsMkkxeTH5AkBX"
            }
        }
    }
    def __init__(self, n_bins, hidden_channels=300, embed_dim=40, num_layers=2, causal=False, rnn_type='lstm', take_log=True, take_db=False, eps=EPS):
        super().__init__()

        self.n_bins = n_bins
        self.hidden_channels, self.embed_dim = hidden_channels, embed_dim
        self.num_layers = num_layers
        self.causal = causal

        self.rnn_type=rnn_type
        self.take_log, self.take_db = take_log, take_db
        self.eps = eps

        if self.take_log and self.take_db:
            raise ValueError("Either take_log or take_db should be False.")

        if causal:
            bidirectional = False
            num_directions = 1
        else:
            bidirectional = True
            num_directions = 2

        self.rnn = choose_rnn(rnn_type, input_size=n_bins, hidden_size=hidden_channels, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Linear(num_directions*hidden_channels, n_bins*embed_dim)
    
    def forward(self, input):
        """
        Args:
            input <torch.Tensor>: Amplitude with shape of (batch_size, 1, n_bins, n_frames).
        Returns:
            output <torch.Tensor>: (batch_size, n_bins, n_frames, embed_dim)
        """
        n_bins, embed_dim = self.n_bins, self.embed_dim
        eps = self.eps

        batch_size, _, _, n_frames = input.size()

        if self.take_log:
            x = torch.log(input + eps)
        elif self.take_db:
            x = 20 * torch.log10(input + eps)
        else:
            x = input

        x = x.squeeze(dim=1).permute(0, 2, 1) # (batch_size, n_frames, n_bins)
        x, _ = self.rnn(x)
        x = self.fc(x) # (batch_size, n_frames, n_bins * embed_dim)
        x = x.view(batch_size, n_frames, n_bins, embed_dim)
        x = x.permute(0, 2, 1, 3).contiguous() # (batch_size, n_bins, n_frames, embed_dim)
        norm = torch.sum(x**2, dim=1, keepdim=True)
        output = x / (norm + eps)

        return output

    def get_config(self):
        config = {
            'n_bins': self.n_bins,
            'embed_dim': self.embed_dim,
            'hidden_channels': self.hidden_channels,
            'num_layers': self.num_layers,
            'causal': self.causal,
            'rnn_type': self.rnn_type,
            'take_log': self.take_log, 'take_db': self.take_db,
            'eps': self.eps
        }
        
        return config
    
    @classmethod
    def build_model(cls, model_path, load_state_dict=False):
        config = torch.load(model_path, map_location=lambda storage, loc: storage)
        
        n_bins = config['n_bins']
        embed_dim = config['embed_dim']
        hidden_channels = config['hidden_channels']
        num_layers = config['num_layers']
        causal = config['causal']

        rnn_type = config['rnn_type']

        take_log, take_db = config['take_log'], config['take_db']
        
        eps = config['eps']
        
        model = cls(
            n_bins, embed_dim=embed_dim, hidden_channels=hidden_channels,
            num_layers=num_layers, causal=causal, rnn_type=rnn_type,
            take_log=take_log, take_db=take_db,
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
        
        if task in ['wsj0-mix', 'wsj0']:
            sample_rate = kwargs.get('sample_rate') or 8000
            n_sources = kwargs.get('n_sources') or 2
            model_choice = kwargs.get('model_choice') or 'last'

            model_id = pretrained_model_ids_task[sample_rate][n_sources]
            download_dir = os.path.join(root, cls.__name__, task, "sr{}/{}speakers".format(sample_rate, n_sources))

            additional_attributes.update({
                'n_sources': n_sources
            })
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

        if task in ['wsj0-mix', 'wsj0']:
            additional_attributes.update({
                'n_fft': config['n_fft'], 'hop_length': config['hop_length'],
                'window_fn': config['window_fn'],
                'threshold': config['threshold']
            })

        for key, value in additional_attributes.items():
            setattr(model, key, value)

        return model
    
    @property
    def num_parameters(self):
        _num_parameters = 0
        
        for p in self.parameters():
            if p.requires_grad:
                _num_parameters += p.numel()
                
        return _num_parameters

class DeepEmbedding_pp(nn.Module):
    def __init__(self, n_bins, hidden_channels=300, embed_dim=40, num_layers=4, enh_hidden_channels=600, enh_num_layers=2, causal=False, rnn_type='lstm', eps=EPS, **kwargs):
        super().__init__()

        self.n_bins = n_bins
        self.hidden_channels, self.embed_dim = hidden_channels, embed_dim

        self.eps = eps

        if causal:
            bidirectional = False
            num_directions = 1
        else:
            bidirectional = True
            num_directions = 2

        self.rnn = choose_rnn(rnn_type, input_size=n_bins, hidden_size=hidden_channels, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_channels * num_directions, n_bins * embed_dim)
        self.embed_nonlinear = nn.Sigmoid()
        self.net_enhancement = NaiveEnhancementNet(2 * n_bins, n_bins, hidden_channels=enh_hidden_channels, num_layers=enh_num_layers, causal=causal, eps=eps)
    
    def forward(self, input):
        """
        Args:
            input (batch_size, n_bins, n_frames): Input feature. This input is expected log-magnitude.
        Returns:
            output (batch_size, embed_dim, n_bins, n_frames): Embedded feature.
        """
        n_bins, embed_dim = self.n_bins, self.embed_dim
        eps = self.eps

        batch_size, _, n_frames = input.size()

        x = input.permute(0, 2, 1).contiguous() # (batch_size, n_frames, n_bins)
        x, _ = self.rnn(x)
        x = self.fc(x) # (batch_size, n_frames, n_bins * embed_dim)
        x = x.view(batch_size, n_frames, n_bins, embed_dim)
        x = x.permute(0, 3, 2, 1).contiguous() # (batch_size, embed_dim, n_bins, n_frames)
        norm = torch.sum(x**2, dim=1, keepdim=True)
        x = x / (norm + eps)
        output = self.embed_nonlinear(x)

        return output
    
    @property
    def num_parameters(self):
        _num_parameters = 0
        
        for p in self.parameters():
            if p.requires_grad:
                _num_parameters += p.numel()
                
        return _num_parameters

class NaiveEnhancementNet(nn.Module):
    def __init__(self, num_features, n_bins, hidden_channels=300, num_layers=2, causal=False, rnn_type='lstm', eps=EPS, **kwargs):
        super().__init__()

        self.eps = eps

        if causal:
            bidirectional = False
            num_directions = 1
        else:
            bidirectional = True
            num_directions = 2
        
        self.rnn = choose_rnn(rnn_type, input_size=num_features, hidden_size=hidden_channels, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_channels * num_directions, n_bins)
        self.nonlinear = nn.Softmax(dim=1)
    
    def forward(self, input):
        return input

class ChimeraNet(nn.Module):
    def __init__(self, n_bins, hidden_channels=300, embed_dim=20, num_layers=2, causal=False, n_sources=2, rnn_type='lstm', eps=EPS, **kwargs):
        super().__init__()

        self.n_bins = n_bins
        self.hidden_channels, self.embed_dim = hidden_channels, embed_dim

        self.eps = eps

        if causal:
            bidirectional = True
            num_directions = 2
        else:
            bidirectional = False
            num_directions = 1

        self.rnn = choose_rnn(rnn_type, input_size=n_bins, hidden_size=hidden_channels, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)

        self.embed_fc = nn.Linear(hidden_channels*num_directions, n_bins*embed_dim)
        self.embed_nonlinear = nn.Tanh()

        self.mask_fc = nn.Linear(hidden_channels, n_bins*n_sources)
        self.mask_nonlinear = nn.Softmax(dim=1)
    
    def forward(self, input):
        """
        Args:
            input (batch_size, n_bins, n_frames): Input feature. This input is expected log-magnitude.
        Returns:
            output (batch_size, embed_dim, n_bins, n_frames): Embedded feature.
        """
        n_bins, embed_dim = self.n_bins, self.embed_dim
        eps = self.eps

        batch_size, _, n_frames = input.size()

        x = input.permute(0, 2, 1).contiguous() # (batch_size, n_frames, n_bins)
        x, _ = self.rnn(x)
        x = self.fc(x) # (batch_size, n_frames, n_bins * embed_dim)
        x = x.view(batch_size, n_frames, n_bins, embed_dim)
        x = x.permute(0, 3, 2, 1).contiguous() # (batch_size, embed_dim, n_bins, n_frames)
        norm = torch.sum(x**2, dim=1, keepdim=True)
        output = x / (norm + eps)

        return output
    
    @property
    def num_parameters(self):
        _num_parameters = 0
        
        for p in self.parameters():
            if p.requires_grad:
                _num_parameters += p.numel()
                
        return _num_parameters

def _test_deep_embedding():
    batch_size, T = 2, 512
    n_sources = 2
    n_fft, hop_length = 256, 128
    window_fn = 'hann'
    n_bins = n_fft // 2 + 1
    hidden_channels, embed_dim = 16, 20

    criterion = AffinityLoss()

    window = build_window(n_fft, window_fn=window_fn)
    waveform = torch.randn((batch_size, n_sources, T), dtype=torch.float)

    sources = stft(waveform, n_fft, hop_length=hop_length, window=window, onesided=True, return_complex=True)
    mixture = sources.sum(dim=1, keepdim=True)
    input = torch.abs(mixture)
    target = compute_ideal_binary_mask(sources, source_dim=1) # (batch_size, n_sources, n_bins, n_frames)

    # Non causal
    print("-"*10, "Non causal", "-"*10)

    model = DeepEmbedding(n_bins, hidden_channels=hidden_channels, embed_dim=embed_dim, causal=False)
    print(model)
    print("# Parameters: {}".format(model.num_parameters))
    
    output = model(input)
    print(input.size(), output.size(), target.size())

    output = output.view(batch_size, -1, embed_dim)
    target = target.view(batch_size, n_sources, -1).permute(0, 2, 1)

    loss = criterion(output, target, batch_mean=False)
    print(loss)

    kmeans = KMeans(K=n_sources)

    kmeans.train()
    cluster_ids = kmeans(output)


def _test_chimeranet():
    pass

if __name__ == '__main__':
    from utils.audio import build_window
    from transforms.stft import stft
    from algorithm.clustering import KMeans
    from algorithm.frequency_mask import compute_ideal_binary_mask
    from criterion.deep_clustering import AffinityLoss

    torch.manual_seed(111)
    
    print("="*10, "Deep embedding", "="*10)
    _test_deep_embedding()
    print()

    print("="*10, "Chimera Net", "="*10)
    _test_chimeranet()