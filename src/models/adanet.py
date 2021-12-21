import itertools

import torch
import torch.nn as nn

from transforms.stft import stft, istft
from models.danet import DANet, DANetTimeDomainWrapper

EPS = 1e-12

"""
    Anchored DANet
"""
class ADANet(DANet):
    pretrained_model_ids = {
        "wsj0-mix": {
            8000: {
                2: "1-02OJ33QlQ_rvgbd4KLX23A5NoShHA-L",
                3: "1-BW-HQtszmnUHRPBPLwY9rjNxLpD9rm0"
            }
        }
    }
    def __init__(self, n_bins, embed_dim=20, hidden_channels=600, num_blocks=4, num_anchors=6, dropout=5e-1, causal=False, mask_nonlinear='sigmoid', take_log=True, take_db=False, permute_anchors=False, eps=EPS, **kwargs):
        super().__init__(n_bins, embed_dim=embed_dim, hidden_channels=hidden_channels, num_blocks=num_blocks, dropout=dropout, causal=causal, mask_nonlinear=mask_nonlinear, eps=eps, take_log=take_log, take_db=take_db, **kwargs)

        self.num_anchors = num_anchors
        self.permute_anchors = permute_anchors
        self.anchor = nn.Parameter(torch.Tensor(num_anchors, embed_dim), requires_grad=True)

        self._reset_parameters()

    def forward(self, input, threshold_weight=None, n_sources=None):
        """
        Args:
            input <torch.Tensor>: Amplitude with shape of (batch_size, 1, n_bins, n_frames).
            assignment <torch.Tensor>: Speaker assignment during training. Tensor shape is (batch_size, n_sources, n_bins, n_frames).
            threshold_weight <torch.Tensor> or <float>: (batch_size, 1, n_bins, n_frames)
        Returns:
            output (batch_size, n_sources, n_bins, n_frames)
        """
        output, _, _ = self.extract_latent(input, threshold_weight=threshold_weight, n_sources=n_sources)

        return output

    def extract_latent(self, input, threshold_weight=None, n_sources=None):
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
        if n_sources is None:
            raise ValueError("Specify n_sources!")

        num_anchors = self.num_anchors
        embed_dim = self.embed_dim
        permute_anchors = self.permute_anchors
        eps = self.eps

        if permute_anchors:
            patterns = list(itertools.permutations(range(num_anchors), n_sources))
        else:
            patterns = list(itertools.combinations(range(num_anchors), n_sources))

        n_patterns = len(patterns)
        patterns = torch.Tensor(patterns).long()
        patterns = patterns.to(self.anchor.device)
        anchor_combination = self.anchor[patterns] # (n_patterns, n_sources, embed_dim)

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

        distance_combination = []

        for anchor in anchor_combination:
            distance = torch.sum(anchor.unsqueeze(dim=1) * latent.unsqueeze(dim=1), dim=-1) # (batch_size, n_sources, n_bins * n_frames)
            distance_combination.append(distance)

        distance_combination = torch.stack(distance_combination, dim=0) # (n_patterns, batch_size, n_sources, n_bins * n_frames)
        assignment_combination = torch.softmax(distance_combination, dim=2) # (n_patterns, batch_size, n_sources, n_bins * n_frames)

        if threshold_weight is not None:
            threshold_weight = threshold_weight.view(1, batch_size, 1, n_bins * n_frames)
            assignment_combination = threshold_weight * assignment_combination # (n_patterns, batch_size, n_sources, n_bins * n_frames)

        attractor_combination, max_similarity_combination = [], []

        for assignment in assignment_combination:
            attractor = torch.bmm(assignment, latent) / (assignment.sum(dim=2, keepdim=True) + eps) # (batch_size, n_sources, embed_dim)
            similarity = torch.bmm(attractor, attractor.permute(0, 2, 1)) # (batch_size, n_sources, n_sources)
            masked_similarity = torch.triu(similarity, diagonal=1)
            masked_similarity = masked_similarity.view(batch_size, n_sources * n_sources) # (batch_size, n_sources * n_sources)
            max_similarity, _ = torch.max(masked_similarity, dim=1) # (batch_size,)

            attractor_combination.append(attractor)
            max_similarity_combination.append(max_similarity)

        attractor_combination = torch.stack(attractor_combination, dim=1) # (batch_size, n_patterns, n_sources, embed_dim)
        flatten_attractor_combination = attractor_combination.view(batch_size * n_patterns, n_sources, embed_dim)
        max_similarity_combination = torch.stack(max_similarity_combination, dim=1) # (batch_size, n_patterns)
        indices = torch.argmin(max_similarity_combination, dim=1) # (batch_size,)
        flatten_indices = indices + torch.arange(0, batch_size * n_patterns, n_patterns).to(indices.device) # (batch_size,)
        flatten_indices = flatten_indices.long()
        attractor = flatten_attractor_combination[flatten_indices] # (batch_size, n_sources, embed_dim)

        similarity = torch.bmm(attractor, latent.permute(0, 2, 1)) # (batch_size, n_sources, n_bins * n_frames)
        similarity = similarity.view(batch_size, n_sources, n_bins, n_frames)
        mask = self.mask_nonlinear2d(similarity) # (batch_size, n_sources, n_bins, n_frames)
        output = mask * input

        latent = latent.view(batch_size, n_bins, n_frames, embed_dim)

        return output, latent, attractor

    def _reset_parameters(self):
        nn.init.orthogonal_(self.anchor.data)

    def get_config(self):
        config = super().get_config()
        config['num_anchors'] = self.num_anchors
        config['permute_anchors'] = self.permute_anchors

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

        num_anchors = config['num_anchors']
        permute_anchors = config.get('permute_anchors', False)

        eps = config['eps']

        model = cls(
            n_bins, embed_dim=embed_dim, hidden_channels=hidden_channels,
            num_blocks=num_blocks, num_anchors=num_anchors,
            dropout=dropout, causal=causal, mask_nonlinear=mask_nonlinear, 
            take_log=take_log, take_db=take_db,
            permute_anchors=permute_anchors,
            eps=eps
        )

        if load_state_dict:
            model.load_state_dict(config['state_dict'])

        return model

    @classmethod
    def build_from_pretrained(cls, root="./pretrained", quiet=False, load_state_dict=True, **kwargs):
        import os

        from utils.utils import download_pretrained_model_from_google_drive

        task = kwargs.get('task')

        if not task in cls.pretrained_model_ids:
            raise KeyError("Invalid task ({}) is specified.".format(task))

        pretrained_model_ids_task = cls.pretrained_model_ids[task]
        additional_attributes = {}

        if task in ['wsj0-mix', 'wsj0']:
            sample_rate = kwargs.get('sample_rate') or 8000
            n_sources = kwargs.get('n_sources') or 2
            model_choice = kwargs.get('model_choice') or "last"

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
    def TimeDomainWrapper(cls, base_model, n_fft, hop_length=None, window_fn='hann'):
        return ADANetTimeDomainWrapper(base_model, n_fft, hop_length=hop_length, window_fn=window_fn)

    @property
    def num_parameters(self):
        _num_parameters = 0

        for p in self.parameters():
            if p.requires_grad:
                _num_parameters += p.numel()

        return _num_parameters

class ADANetTimeDomainWrapper(DANetTimeDomainWrapper):
    def __init__(self, base_model: ADANet, n_fft, hop_length=None, window_fn='hann', eps=EPS):
        super().__init__(base_model, n_fft, hop_length=hop_length, window_fn=window_fn, eps=eps)

    def forward(self, input, threshold=None, n_sources=None):
        """
        Args:
            input <torch.Tensor>: (batch_size, 1, T)
            threshold <float>: threshold [dB]
            n_sources <int>: Number of sources
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

        estimated_amplitude = self.base_model(mixture_amplitude, threshold_weight=threshold_weight, n_sources=n_sources)
        estimated_spectrogram = estimated_amplitude * torch.exp(1j * mixture_angle)
        output = istft(estimated_spectrogram, self.n_fft, hop_length=self.hop_length, window=self.window, onesided=True, return_complex=False, length=T)

        return output

def _test_adanet():
    torch.manual_seed(111)

    batch_size = 2
    N = 6
    K = 10
    H = 32
    B = 4

    n_bins, n_frames = 4, 128
    n_sources = 2
    causal = False
    mask_nonlinear = 'sigmoid'

    sources = torch.randn((batch_size, n_sources, n_bins, n_frames), dtype=torch.float)
    input = sources.sum(dim=1, keepdim=True)
    threshold_weight = torch.randint(0, 2, (batch_size, 1, n_bins, n_frames), dtype=torch.float)

    model = ADANet(n_bins, embed_dim=K, hidden_channels=H, num_blocks=B, num_anchors=N, causal=causal, mask_nonlinear=mask_nonlinear)
    print(model)
    print("# Parameters: {}".format(model.num_parameters))

    output = model(input, threshold_weight=threshold_weight, n_sources=n_sources)

    print(input.size(), output.size())

def _test_adanet_paper():
    batch_size = 2
    N = 6
    K = 20
    H = 300
    B = 4

    n_bins, n_frames = 129, 256
    n_sources = 2
    causal = False
    mask_nonlinear = 'sigmoid'

    sources = torch.randn((batch_size, n_sources, n_bins, n_frames), dtype=torch.float)
    input = sources.sum(dim=1, keepdim=True)
    threshold_weight = torch.randint(0, 2, (batch_size, 1, n_bins, n_frames), dtype=torch.float)

    model = ADANet(n_bins, embed_dim=K, hidden_channels=H, num_blocks=B, num_anchors=N, causal=causal, mask_nonlinear=mask_nonlinear)
    print(model)
    print("# Parameters: {}".format(model.num_parameters))

    output = model(input, threshold_weight=threshold_weight, n_sources=n_sources)

    print(input.size(), output.size())

if __name__ == '__main__':
    torch.manual_seed(111)

    print("="*10, "ADANet", "="*10)
    _test_adanet()
    print()

    print("="*10, "ADANet (paper)", "="*10)
    _test_adanet_paper()
