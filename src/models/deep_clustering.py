import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.audio import build_window
from utils.model import choose_rnn
from algorithm.clustering import KMeans
from transforms.stft import stft, istft

EPS = 1e-12

class DeepEmbedding(nn.Module):
    pretrained_model_ids = {
        "wsj0-mix": {
            8000: {
                2: "111Q6FLpLXSahK3YVO0m0JE5XieYLBsG4",
                3: "1-27Q01Ie5K3dezaUv9AYTnQ9Xxr2dF87"
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
        self.fc = nn.Linear(num_directions * hidden_channels, n_bins*embed_dim)

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
        norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
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

    @classmethod
    def TimeDomainWrapper(cls, base_model, n_fft, hop_length=None, window_fn='hann', eps=EPS):
        return DeepEmbeddingTimeDomainWrapper(base_model, n_fft, hop_length=hop_length, window_fn=window_fn, eps=eps)

    @property
    def num_parameters(self):
        _num_parameters = 0

        for p in self.parameters():
            if p.requires_grad:
                _num_parameters += p.numel()

        return _num_parameters

class DeepEmbeddingTimeDomainWrapper(nn.Module):
    def __init__(self, base_model: DeepEmbedding, n_fft, hop_length=None, window_fn='hann', eps=EPS):
        super().__init__()

        self.base_model = base_model

        if hop_length is None:
            hop_length = n_fft // 4

        self.n_fft, self.hop_length = n_fft, hop_length
        window = build_window(n_fft, window_fn=window_fn)
        self.register_buffer("window", window)

        self.eps = eps

    def forward(self, input, threshold=None, n_sources=None, iter_clustering=None):
        """
        Args:
            input <torch.Tensor>: (batch_size, 1, T)
            threshold <float>: threshold [dB]
            n_sources <int>: Number of sources
            iter_clustering <int>: Number of iteration in KMeans
        Returns:
            output <torch.Tensor>: (batch_size, n_sources, T)
        """
        assert input.dim() == 3, "input is expected 3D input."

        T = input.size(-1)

        mixture_spectrogram = stft(input, self.n_fft, hop_length=self.hop_length, window=self.window, onesided=True, return_complex=True)
        mixture_amplitude, mixture_angle = torch.abs(mixture_spectrogram), torch.angle(mixture_spectrogram)

        batch_size, _, n_bins, n_frames = mixture_spectrogram.size()

        if threshold is not None:
            log_amplitude = 20 * torch.log10(mixture_amplitude + self.eps)
            max_log_amplitude = torch.max(log_amplitude)
            threshold = 10**((max_log_amplitude - threshold) / 20)
            threshold_weight = torch.where(mixture_amplitude > threshold, torch.ones_like(mixture_amplitude), torch.zeros_like(mixture_amplitude))
        else:
            threshold_weight = None

        latent = self.base_model(mixture_amplitude)
        latent = latent.view(batch_size, n_bins * n_frames, latent.size(-1))

        if threshold_weight is not None:
            assert batch_size == 1, "KMeans is expected same number of samples among all batches, so if threshold_weight is given, batch_size should be 1."

            latent = latent.squeeze(dim=0) # (n_bins * n_frames, embed_dim)
            salient_indices, = torch.nonzero(threshold_weight.flatten(), as_tuple=True)
            latent_salient = latent[salient_indices]

            kmeans = KMeans(K=n_sources)
            kmeans.train()
            _ = kmeans(latent_salient)

            kmeans.eval()
            cluster_ids = kmeans(latent, iteration=iter_clustering) # (n_bins * n_frames,)
        else:
            kmeans = KMeans(K=n_sources)
            kmeans.train()
            cluster_ids = kmeans(latent, iteration=iter_clustering) # (batch_size, n_bins * n_frames)

        cluster_ids = cluster_ids.view(batch_size, n_bins, n_frames) # (batch_size, n_bins, n_frames)
        mask = torch.eye(n_sources)[cluster_ids] # (batch_size, n_bins, n_frames, n_sources)
        mask = mask.permute(0, 3, 1, 2).contiguous() # (batch_size, n_sources, n_bins, n_frames)
        estimated_amplitude = mask * mixture_amplitude
        estimated_spectrogram = estimated_amplitude * torch.exp(1j * mixture_angle)
        output = istft(estimated_spectrogram, self.n_fft, hop_length=self.hop_length, window=self.window, onesided=True, return_complex=False, length=T)

        return output

class DeepEmbeddingPlus(nn.Module):
    def __init__(self, embedding_net, enhancement_net):
        super().__init__()

        self.embedding_net = embedding_net
        self.enhancement_net = enhancement_net

    def forward(self, input, n_sources=None, iter_clustering=None):
        """
        Args:
            input <torch.Tensor>: Amplitude with shape of (batch_size, 1, n_bins, n_frames).
        Returns:
            embedding <torch.Tensor>: (batch_size, n_bins, n_frames, embed_dim)
            s_tilde <torch.Tensor>: (batch_size, n_sources, n_bins, n_frames)
        """
        assert n_sources is None

        batch_size, _, n_bins, n_frames = input.size()

        embedding = self.embedding_net(input) # (batch_size, n_bins, n_frames, embed_dim)
        embedding_reshaped = embedding.view(batch_size, n_bins * n_frames, -1)

        kmeans = KMeans(K=n_sources)
        kmeans.train()
        cluster_ids = kmeans(embedding_reshaped, iteration=iter_clustering) # (batch_size, n_bins * n_frames)
        mask = F.one_hot(cluster_ids, classes=n_sources) # (batch_size, n_bins * n_frames, n_sources)
        mask = mask.view(batch_size, n_bins, n_frames, n_sources)
        mask = mask.permute(0, 3, 1, 2) # (batch_size, n_sources, n_bins, n_frames)

        s_hat = mask * input # (batch_size, n_sources, n_bins, n_frames)
        s_hat = s_hat.view(batch_size * n_sources, 1, n_bins, n_frames)
        repeated_input = torch.tile(input, (1, n_sources, 1, 1))
        repeated_input = repeated_input.view(batch_size * n_sources, 1, n_bins, n_frames)
        x = torch.cat([repeated_input, s_hat], dim=1) # (batch_size * n_sources, 2, n_bins, n_frames)

        x = self.enhancement_net(x) # (batch_size * n_sources, n_bins, n_frames)
        x = x.view(batch_size, n_sources, n_bins, n_frames)
        mask_final = F.softmax(x, dim=1) # (batch_size, n_sources, n_bins, n_frames)
        s_tilde = mask_final * input

        return embedding, s_tilde

    @property
    def num_parameters(self):
        _num_parameters = 0

        for p in self.parameters():
            if p.requires_grad:
                _num_parameters += p.numel()

        return _num_parameters

class DeepEmbedding_pp(nn.Module):
    def __init__(self, n_bins, hidden_channels=300, embed_dim=40, num_layers=4, enh_hidden_channels=600, enh_num_layers=2, causal=False, rnn_type='lstm', take_log=True, take_db=False, eps=EPS, **kwargs):
        super().__init__()

        self.n_bins = n_bins
        self.hidden_channels, self.embed_dim = hidden_channels, embed_dim

        self.rnn_type = rnn_type
        self.take_log, self.take_db = take_log, take_db
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
        self.enhancement_net = NaiveEnhancementNet(2 * n_bins, n_bins, hidden_channels=enh_hidden_channels, num_layers=enh_num_layers, causal=causal, eps=eps)

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
        norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
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

DeepClustering = DeepEmbedding

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
    from algorithm.frequency_mask import compute_ideal_binary_mask
    from criterion.deep_clustering import AffinityLoss

    torch.manual_seed(111)

    print("="*10, "Deep embedding", "="*10)
    _test_deep_embedding()
    print()

    print("="*10, "Chimera Net", "="*10)
    _test_chimeranet()