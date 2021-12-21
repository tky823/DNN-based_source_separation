import os

import torch
import torch.nn as nn

from utils.audio import build_window
from utils.model import choose_nonlinear
from algorithm.clustering import KMeans
from transforms.stft import stft, istft

SAMPLE_RATE_LIBRISPEECH = 16000
EPS = 1e-12

class DANet(nn.Module):
    pretrained_model_ids = {
        "wsj0-mix": {
            8000: {
                2: "1PTBTUpz5DUZazQRWzhAknYfcUSS76SkI",
                3: "1-3bjp3Dm44CwWiJ36efS7wpIai5Bl95h"
            }
        },
        "librispeech": {
            SAMPLE_RATE_LIBRISPEECH: {
                2: "18FJrUHawpxsJovgb26V8IuHZ5gannwQm"
            }
        }
    }
    def __init__(self, n_bins, embed_dim=20, hidden_channels=300, num_blocks=4, dropout=0, causal=False, mask_nonlinear='sigmoid', take_log=True, take_db=False, eps=EPS):
        super().__init__()

        self.n_bins = n_bins
        self.hidden_channels, self.embed_dim = hidden_channels, embed_dim
        self.num_blocks = num_blocks

        self.dropout = dropout
        self.causal = causal

        if causal:
            num_directions = 1
            bidirectional = False
        else:
            num_directions = 2
            bidirectional = True

        self.mask_nonlinear = mask_nonlinear

        self.rnn = nn.LSTM(n_bins, hidden_channels, num_layers=num_blocks, batch_first=True, bidirectional=bidirectional, dropout=dropout)
        self.fc = nn.Linear(num_directions*hidden_channels, n_bins*embed_dim)

        kwargs = {}

        if mask_nonlinear == 'softmax':
            kwargs["dim"] = 1

        self.mask_nonlinear2d = choose_nonlinear(mask_nonlinear, **kwargs)

        self.take_log, self.take_db = take_log, take_db
        self.eps = eps

        if self.take_log and self.take_db:
            raise ValueError("Either take_log or take_db should be False.")

    def forward(self, input, assignment=None, threshold_weight=None, n_sources=None, iter_clustering=None):
        """
        Args:
            input <torch.Tensor>: Amplitude with shape of (batch_size, 1, n_bins, n_frames).
            assignment <torch.Tensor>: Speaker assignment during training. Tensor shape is (batch_size, n_sources, n_bins, n_frames).
            threshold_weight <torch.Tensor> or <float>: (batch_size, 1, n_bins, n_frames)
        Returns:
            output <torch.Tensor>: (batch_size, n_sources, n_bins, n_frames)
        """
        output, _, _ = self.extract_latent(input, assignment, threshold_weight=threshold_weight, n_sources=n_sources, iter_clustering=iter_clustering)

        return output

    def extract_latent(self, input, assignment=None, threshold_weight=None, n_sources=None, iter_clustering=None):
        """
        Args:
            input <torch.Tensor>: Amplitude with shape of (batch_size, 1, n_bins, n_frames).
            assignment <torch.Tensor>: Speaker assignment during training. Tensor shape is (batch_size, n_sources, n_bins, n_frames).
            threshold_weight <torch.Tensor> or <float>: (batch_size, 1, n_bins, n_frames)
        Returns:
            output <torch.Tensor>: (batch_size, n_sources, n_bins, n_frames)
            latent <torch.Tensor>: (batch_size, n_bins, n_frames, embed_dim)
            attractor <torch.Tensor>: (batch_size, n_sources, embed_dim)
        """
        if n_sources is not None:
            if assignment is not None and n_sources != assignment.size(1):
                raise ValueError("n_sources is different from assignment.size(1)")
        else:
            if assignment is None:
                raise ValueError("Specify assignment, given None!")
            n_sources = assignment.size(1)

        embed_dim = self.embed_dim
        eps = self.eps

        batch_size, _, n_bins, n_frames = input.size()

        self.rnn.flatten_parameters()

        if self.take_log:
            x = torch.log(input + eps)
        elif self.take_db:
            x = 20 * torch.log10(input + eps)
        else:
            x = input

        x = x.squeeze(dim=1).permute(0, 2, 1).contiguous() # (batch_size, n_frames, n_bins)
        x, _ = self.rnn(x) # (batch_size, n_frames, n_bins)
        x = self.fc(x) # (batch_size, n_frames, embed_dim * n_bins)
        x = x.view(batch_size, n_frames, embed_dim, n_bins)
        x = x.permute(0, 2, 3, 1).contiguous()  # (batch_size, embed_dim, n_bins, n_frames)
        latent = x.view(batch_size, embed_dim, n_bins * n_frames)
        latent = latent.permute(0, 2, 1).contiguous() # (batch_size, n_bins * n_frames, embed_dim)

        if assignment is None:
            if self.training:
                raise ValueError("assignment is required.")

            if threshold_weight is not None:
                assert batch_size == 1, "KMeans is expected same number of samples among all batches, so if threshold_weight is given, batch_size should be 1."

                flatten_latent = latent.view(batch_size * n_bins * n_frames, embed_dim) # (batch_size * n_bins * n_frames, embed_dim)
                flatten_threshold_weight = threshold_weight.view(-1) # (batch_size * n_bins * n_frames)
                nonzero_indices, = torch.nonzero(flatten_threshold_weight, as_tuple=True) # (n_nonzeros,)
                latent_nonzero = flatten_latent[nonzero_indices] # (n_nonzeros, embed_dim)
                latent_nonzero = latent_nonzero.view(batch_size, -1, embed_dim) # (batch_size, n_nonzeros, embed_dim)

            kmeans = KMeans(K=n_sources)
            _ = kmeans(latent, iteration=iter_clustering) # (batch_size, n_bins * n_frames)
            attractor = kmeans.centroids # (batch_size, n_sources, embed_dim)
        else:
            threshold_weight = threshold_weight.view(batch_size, 1, n_bins * n_frames)
            assignment = assignment.view(batch_size, n_sources, n_bins * n_frames) # (batch_size, n_sources, n_bins * n_frames)
            assignment = threshold_weight * assignment
            attractor = torch.bmm(assignment, latent) / (assignment.sum(dim=2, keepdim=True) + eps) # (batch_size, n_sources, embed_dim)

        similarity = torch.bmm(attractor, latent.permute(0, 2, 1)) # (batch_size, n_sources, n_bins * n_frames)
        similarity = similarity.view(batch_size, n_sources, n_bins, n_frames)
        mask = self.mask_nonlinear2d(similarity) # (batch_size, n_sources, n_bins, n_frames)
        output = mask * input

        latent = latent.view(batch_size, n_bins, n_frames, embed_dim)

        return output, latent, attractor

    def extract_latent_by_attractor(self, input, attractor):
        """
        Args:
            input <torch.Tensor>: Amplitude with shape of (batch_size, 1, n_bins, n_frames).
            attractor <torch.Tensor>: Attractor with shape of (n_sources, embed_dim).
        Returns:
            output <torch.Tensor>: (batch_size, n_sources, n_bins, n_frames)
            latent <torch.Tensor>: (batch_size, n_bins, n_frames, embed_dim)
        """
        eps = self.eps

        batch_size, _, n_bins, n_frames = input.size()
        n_sources, embed_dim = attractor.size()

        self.rnn.flatten_parameters()

        if self.take_log:
            x = torch.log(input + eps)
        elif self.take_db:
            x = 20 * torch.log10(input + eps)
        else:
            x = input

        x = x.squeeze(dim=1).permute(0, 2, 1).contiguous() # (batch_size, n_frames, n_bins)
        x, _ = self.rnn(x) # (batch_size, n_frames, n_bins)
        x = self.fc(x) # (batch_size, n_frames, embed_dim * n_bins)
        x = x.view(batch_size, n_frames, embed_dim, n_bins)
        x = x.permute(0, 2, 3, 1).contiguous()  # (batch_size, embed_dim, n_bins, n_frames)
        latent = x.view(batch_size, embed_dim, n_bins * n_frames)
        latent = latent.permute(0, 2, 1).contiguous() # (batch_size, n_bins * n_frames, embed_dim)

        batch_fixed_attractor = attractor.expand(batch_size, -1, -1)
        similarity = torch.bmm(batch_fixed_attractor, latent.permute(0, 2, 1)) # (batch_size, n_sources, n_bins * n_frames)
        similarity = similarity.view(batch_size, n_sources, n_bins, n_frames)
        mask = self.mask_nonlinear2d(similarity) # (batch_size, n_sources, n_bins, n_frames)
        output = mask * input

        latent = latent.view(batch_size, n_bins, n_frames, embed_dim)

        return output, latent

    def get_config(self):
        config = {
            'n_bins': self.n_bins,
            'embed_dim': self.embed_dim,
            'hidden_channels': self.hidden_channels,
            'num_blocks': self.num_blocks,
            'dropout': self.dropout,
            'causal': self.causal,
            'mask_nonlinear': self.mask_nonlinear,
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
        num_blocks = config['num_blocks']
        dropout = config['dropout']

        causal = config['causal']
        mask_nonlinear = config['mask_nonlinear']
        take_log, take_db = config['take_log'], config['take_db']

        eps = config['eps']

        model = cls(
            n_bins, embed_dim=embed_dim, hidden_channels=hidden_channels,
            num_blocks=num_blocks, dropout=dropout, causal=causal, mask_nonlinear=mask_nonlinear,
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
        elif task in ['librispeech']:
            sample_rate = kwargs.get('sample_rate') or SAMPLE_RATE_LIBRISPEECH
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

        if task in ['wsj0-mix', 'wsj0', 'librispeech']:
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
        return DANetTimeDomainWrapper(base_model, n_fft, hop_length=hop_length, window_fn=window_fn, eps=eps)

    @property
    def num_parameters(self):
        _num_parameters = 0

        for p in self.parameters():
            if p.requires_grad:
                _num_parameters += p.numel()

        return _num_parameters

class DANetTimeDomainWrapper(nn.Module):
    def __init__(self, base_model: DANet, n_fft, hop_length=None, window_fn='hann', eps=EPS):
        super().__init__()

        self.base_model = base_model

        if hop_length is None:
            hop_length = n_fft // 4
        
        self.n_fft, self.hop_length = n_fft, hop_length
        window = build_window(n_fft, window_fn=window_fn)
        self.window = nn.Parameter(window, requires_grad=False)

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

        if threshold is not None:
            log_amplitude = 20 * torch.log10(mixture_amplitude + self.eps)
            max_log_amplitude = torch.max(log_amplitude)
            threshold = 10**((max_log_amplitude - threshold) / 20)
            threshold_weight = torch.where(mixture_amplitude > threshold, torch.ones_like(mixture_amplitude), torch.zeros_like(mixture_amplitude))
        else:
            threshold_weight = None

        estimated_amplitude = self.base_model(mixture_amplitude, threshold_weight=threshold_weight, n_sources=n_sources, iter_clustering=iter_clustering)
        estimated_spectrogram = estimated_amplitude * torch.exp(1j * mixture_angle)
        output = istft(estimated_spectrogram, self.n_fft, hop_length=self.hop_length, window=self.window, onesided=True, return_complex=False, length=T)

        return output

class FixedAttractorDANet(nn.Module):
    pretrained_attractor_ids = {
        "wsj0-mix": {
            8000: {
                2: "1-eV-9ciO4toLTWlez63bfBB3jVZiqPRz",
                3: "1-TR6itD1EdU1VKHpweuMjw6SgijSuepF"
            }
        }
    }
    def __init__(self, base_model: DANet, fixed_attractor=None):
        """
        Args:
            base_model <DANet>: Base Deep Attractor Network.
            attractor <torch.Tensor>: Pretrained attractor with shape of (n_sources, embed_dim).
        """
        super().__init__()

        self.base_model = base_model
        self.fixed_attractor = nn.Parameter(fixed_attractor, requires_grad=False)

    def forward(self, input):
        """
        Args:
            input <torch.Tensor>: Amplitude with shape of (batch_size, 1, n_bins, n_frames).
        Returns:
            output <torch.Tensor>: (batch_size, n_sources, n_bins, n_frames)
        """
        output, _ = self.extract_latent(input)

        return output

    def extract_latent(self, input):
        """
        Args:
            input <torch.Tensor>: Amplitude with shape of (batch_size, 1, n_bins, n_frames).
        Returns:
            output <torch.Tensor>: (batch_size, n_sources, n_bins, n_frames)
            latent <torch.Tensor>: (batch_size, n_bins, n_frames, embed_dim)
        """
        output, latent = self.base_model.extract_latent_by_attractor(input, self.fixed_attractor)

        return output, latent

    def get_config(self):
        config = self.base_model.get_config()
        config["attractor_size"] = self.fixed_attractor.size()

        return config

    @classmethod
    def build_model(cls, model_path, load_state_dict=False):
        config = torch.load(model_path, map_location=lambda storage, loc: storage)
        base_model = DANet.build_model(model_path, load_state_dict=False)
        dummy_attractor = torch.empty(*config["attractor_size"])

        model = cls(base_model, dummy_attractor)

        if load_state_dict:
            model.load_state_dict(config['state_dict'])
        else:
            raise ValueError("Set load_state_dict=True")

        return model

    @classmethod
    def build_from_pretrained(cls, root="./pretrained", quiet=False, load_state_dict=True, **kwargs):
        from utils.utils import download_pretrained_model_from_google_drive

        # For pretrained FixedAttractorDANet, pretrained attractor is saved separately from state dict of base DANet,
        base_model = DANet.build_from_pretrained(root, quiet=quiet, load_state_dict=load_state_dict, **kwargs)

        task = kwargs.get('task')

        if not task in cls.pretrained_attractor_ids:
            raise KeyError("Invalid task ({}) is specified.".format(task))

        pretrained_attractor_ids_task = cls.pretrained_attractor_ids[task]
        additional_attributes = {}

        if task in ['wsj0-mix', 'wsj0']:
            sample_rate = kwargs.get('sample_rate') or 8000
            n_sources = kwargs.get('n_sources') or 2
            model_choice = kwargs.get('model_choice') or 'last'

            attractor_id = pretrained_attractor_ids_task[sample_rate][n_sources]
            download_dir = os.path.join(root, cls.__name__, task, "sr{}/{}speakers".format(sample_rate, n_sources))

            additional_attributes.update({
                'n_sources': n_sources
            })
        else:
            raise NotImplementedError("Not support task={}.".format(task))

        additional_attributes.update({
            'sample_rate': sample_rate
        })

        attractor_path = os.path.join(download_dir, "attractor", "{}.pth".format(model_choice))

        if not os.path.exists(attractor_path):
            download_pretrained_model_from_google_drive(attractor_id, download_dir, quiet=quiet)

        if load_state_dict:
            fixed_attractor = torch.load(attractor_path, map_location=lambda storage, loc: storage)
            model = cls(base_model, fixed_attractor)
        else:
            raise ValueError("Set load_state_dict=True")

        if task in ['wsj0-mix', 'wsj0']:
            additional_attributes.update({
                'n_fft': base_model.n_fft, 'hop_length': base_model.hop_length,
                'window_fn': base_model.window_fn
            })

        for key, value in additional_attributes.items():
            setattr(model, key, value)

        return model

    @classmethod
    def TimeDomainWrapper(cls, base_model, n_fft, hop_length=None, window_fn='hann'):
        return FixedAttractorDANetTimeDomainWrapper(base_model, n_fft, hop_length=hop_length, window_fn=window_fn)

    @property
    def num_parameters(self):
        _num_parameters = 0

        for p in self.parameters():
            if p.requires_grad:
                _num_parameters += p.numel()

        return _num_parameters

class FixedAttractorDANetTimeDomainWrapper(nn.Module):
    def __init__(self, base_model: FixedAttractorDANet, n_fft, hop_length=None, window_fn='hann'):
        super().__init__()

        self.base_model = base_model

        if hop_length is None:
            hop_length = n_fft // 4

        self.n_fft, self.hop_length = n_fft, hop_length
        window = build_window(n_fft, window_fn=window_fn)
        self.window = nn.Parameter(window, requires_grad=False)

    def forward(self, input):
        """
        Args:
            input <torch.Tensor>: (batch_size, 1, T)
        Returns:
            output <torch.Tensor>: (batch_size, n_sources, T)
        """
        assert input.dim() == 3, "input is expected 3D input."

        T = input.size(-1)

        mixture_spectrogram = stft(input, self.n_fft, hop_length=self.hop_length, window=self.window, onesided=True, return_complex=True)
        mixture_amplitude, mixture_angle = torch.abs(mixture_spectrogram), torch.angle(mixture_spectrogram)

        estimated_amplitude = self.base_model(mixture_amplitude)
        estimated_spectrogram = estimated_amplitude * torch.exp(1j * mixture_angle)
        output = istft(estimated_spectrogram, self.n_fft, hop_length=self.hop_length, window=self.window, onesided=True, return_complex=False, length=T)

        return output

def _test_danet():
    batch_size = 2
    K = 10

    H = 32
    B = 4

    n_bins, n_frames = 4, 128
    n_sources = 2
    causal = False
    mask_nonlinear = 'sigmoid'

    sources = torch.randn((batch_size, n_sources, n_bins, n_frames), dtype=torch.float)
    input = sources.sum(dim=1, keepdim=True)
    assignment = compute_ideal_binary_mask(sources, source_dim=1)
    threshold_weight = torch.randint(0, 2, (batch_size, 1, n_bins, n_frames), dtype=torch.float)

    model = DANet(n_bins, embed_dim=K, hidden_channels=H, num_blocks=B, causal=causal, mask_nonlinear=mask_nonlinear)
    print(model)
    print("# Parameters: {}".format(model.num_parameters))

    output = model(input, assignment, threshold_weight=threshold_weight)

    print(input.size(), output.size())

def _test_danet_paper():
    batch_size = 2
    K = 20

    H = 300
    B = 4

    n_bins, n_frames = 129, 256
    n_sources = 2
    causal = False
    mask_nonlinear = 'sigmoid'

    sources = torch.randn((batch_size, n_sources, n_bins, n_frames), dtype=torch.float)
    input = sources.sum(dim=1, keepdim=True)
    assignment = compute_ideal_binary_mask(sources, source_dim=1)
    threshold_weight = torch.randint(0, 2, (batch_size, 1, n_bins, n_frames), dtype=torch.float)

    model = DANet(n_bins, embed_dim=K, hidden_channels=H, num_blocks=B, causal=causal, mask_nonlinear=mask_nonlinear)
    print(model)
    print("# Parameters: {}".format(model.num_parameters))

    output = model(input, assignment, threshold_weight=threshold_weight)

    print(input.size(), output.size())

def _test_fixed_attractor_danet():
    batch_size = 2
    K = 10

    H = 32
    B = 4

    n_bins, n_frames = 4, 128
    n_sources = 2
    causal = False
    mask_nonlinear = 'sigmoid'

    sources = torch.randn((batch_size, n_sources, n_bins, n_frames), dtype=torch.float)
    input = sources.sum(dim=1, keepdim=True)
    attractor = torch.randn(n_sources, K)

    base_model = DANet(n_bins, embed_dim=K, hidden_channels=H, num_blocks=B, causal=causal, mask_nonlinear=mask_nonlinear)
    model = FixedAttractorDANet(base_model, fixed_attractor=attractor)

    print(model)
    print("# Parameters: {}".format(model.num_parameters))

    output = model(input)

    print(input.size(), output.size())

if __name__ == '__main__':
    from algorithm.frequency_mask import compute_ideal_binary_mask

    torch.manual_seed(111)

    print("="*10, "DANet", "="*10)
    _test_danet()
    print()

    print("="*10, "DANet (same configuration in paper)", "="*10)
    _test_danet_paper()
    print()

    print("="*10, "DANet w/ fixed attractor", "="*10)
    _test_fixed_attractor_danet()