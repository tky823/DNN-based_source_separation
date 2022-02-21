import os

import yaml
import torch
import torch.nn as nn

from utils.audio import build_window
from algorithm.frequency_mask import multichannel_wiener_filter
from transforms.stft import stft, istft
from models.mm_dense_rnn import MMDenseRNN

__sources__ = ['bass', 'drums', 'other', 'vocals']
FULL = 'full'
SAMPLE_RATE_MUSDB18 = 44100
EPS = 1e-12

class ParallelMMDenseLSTM(nn.Module):
    def __init__(self, modules):
        super().__init__()

        if isinstance(modules, nn.ModuleDict):
            pass
        elif isinstance(modules, dict):
            modules = nn.ModuleDict(modules)
        else:
            raise TypeError("Type of `modules` is expected nn.ModuleDict or dict, but given {}.".format(type(modules)))

        in_channels = None
        sources = list(modules.keys())

        for key in sources:
            module = modules[key]
            if not isinstance(module, MMDenseLSTM):
                raise ValueError("All modules must be MMDenseLSTM.")

            if in_channels is None:
                in_channels = module.in_channels
            else:
                assert in_channels == module.in_channels, "`in_channels` are different among modules."

        self.net = modules

        self.in_channels = in_channels
        self.sources = sources

    def forward(self, input, target=None):
        """
        Args:
            input: Nonnegative tensor with shape of
                (batch_size, in_channels, n_bins, n_frames) if target is specified.
                (batch_size, 1, in_channels, n_bins, n_frames) if target is None.
        Returns:
            output:
                (batch_size, in_channels, n_bins, n_frames) if target is specified.
                (batch_size, n_sources, in_channels, n_bins, n_frames) if target is None.
        """
        if target is None:
            assert input.dim() == 5, "input is expected 5D, but given {}.".format(input.dim())
            input = input.squeeze(dim=1)
            output = []
            for target in self.sources:
                _output = self.net[target](input)
                output.append(_output)
            output = torch.stack(output, dim=1)
        else:
            if type(target) is not str:
                raise TypeError("`target` is expected str, but given {}".format(type(target)))

            assert input.dim() == 4, "input is expected 4D, but given {}.".format(input.dim())

            output = self.net[target](input)

        return output

    @classmethod
    def build_from_pretrained(cls, root="./pretrained", quiet=False, load_state_dict=True, **kwargs):        
        from utils.utils import download_pretrained_model_from_google_drive

        task = kwargs.get('task')

        if not task in MMDenseLSTM.pretrained_model_ids:
            raise KeyError("Invalid task ({}) is specified.".format(task))

        pretrained_model_ids_task = MMDenseLSTM.pretrained_model_ids[task]
        additional_attributes = {}

        if task in ['musdb18', 'musdb18hq']:
            sample_rate = kwargs.get('sample_rate') or SAMPLE_RATE_MUSDB18
            config = kwargs.get('config') or "paper"
            sources = __sources__
            model_choice = kwargs.get('model_choice') or 'best'

            model_id = pretrained_model_ids_task[sample_rate][config]
            download_dir = os.path.join(root, MMDenseLSTM.__name__, task, "sr{}".format(sample_rate), config)
        else:
            raise NotImplementedError("Not support task={}.".format(task))

        additional_attributes.update({
            'sample_rate': sample_rate
        })

        modules = {}
        n_fft, hop_length = None, None
        window_fn = None

        for target in sources:
            model_path = os.path.join(download_dir, "model", target, "{}.pth".format(model_choice))

            if not os.path.exists(model_path):
                download_pretrained_model_from_google_drive(model_id, download_dir, quiet=quiet)

            config = torch.load(model_path, map_location=lambda storage, loc: storage)
            modules[target] = MMDenseLSTM.build_model(model_path, load_state_dict=load_state_dict)

            if task in ['musdb18', 'musdb18hq']:
                if n_fft is None:
                    n_fft = config['n_fft']
                else:
                    assert n_fft == config['n_fft'], "`n_fft` is different among models."

                if hop_length is None:
                    hop_length = config['hop_length']
                else:
                    assert hop_length == config['hop_length'], "`hop_length` is different among models."

                if window_fn is None:
                    window_fn = config['window_fn']
                else:
                    assert window_fn == config['window_fn'], "`window_fn` is different among models."

        additional_attributes.update({
            'n_fft': n_fft, 'hop_length': hop_length,
            'window_fn': window_fn,
        })

        model = cls(modules)

        for key, value in additional_attributes.items():
            setattr(model, key, value)

        return model

    @classmethod
    def TimeDomainWrapper(cls, base_model, n_fft, hop_length=None, window_fn='hann', eps=EPS):
        return ParallelMMDenseLSTMTimeDomainWrapper(base_model, n_fft, hop_length=hop_length, window_fn=window_fn, eps=eps)

    @property
    def num_parameters(self):
        _num_parameters = 0

        for p in self.parameters():
            if p.requires_grad:
                _num_parameters += p.numel()

        return _num_parameters

class ParallelMMDenseLSTMTimeDomainWrapper(nn.Module):
    def __init__(self, base_model: ParallelMMDenseLSTM, n_fft, hop_length=None, window_fn='hann', eps=EPS):
        super().__init__()

        self.base_model = base_model

        if hop_length is None:
            hop_length = n_fft // 4

        self.n_fft, self.hop_length = n_fft, hop_length
        window = build_window(n_fft, window_fn=window_fn)
        self.window = nn.Parameter(window, requires_grad=False)

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

        estimated_amplitude = []

        for target in self.sources:
            _estimated_amplitude = self.base_model(mixture_amplitude.squeeze(dim=1), target=target)
            estimated_amplitude.append(_estimated_amplitude)

        estimated_amplitude = torch.stack(estimated_amplitude, dim=1)
        estimated_spectrogram = multichannel_wiener_filter(mixture_spectrogram, estimated_sources_amplitude=estimated_amplitude, iteration=iteration, eps=eps)
        output = istft(estimated_spectrogram, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window, onesided=True, return_complex=False, length=T)

        return output

    @property
    def sources(self):
        return list(self.base_model.sources)

class MMDenseLSTM(MMDenseRNN):
    pretrained_model_ids = {
        "musdb18": {
            SAMPLE_RATE_MUSDB18: {
                "paper": "1-2JGWMgVBdSj5zF9hl27jKhyX7GN-cOV"
            }
        },
    }
    def __init__(
        self,
        in_channels, num_features,
        growth_rate, hidden_channels,
        kernel_size,
        bands=['low','middle','high'], sections=[380,644,1025],
        scale=(2,2),
        dilated=False, norm=True, nonlinear='relu',
        depth=None,
        growth_rate_final=None, hidden_channels_final=None,
        kernel_size_final=None,
        dilated_final=False,
        norm_final=True, nonlinear_final='relu',
        depth_final=None,
        causal=False,
        rnn_position='parallel',
        eps=EPS,
        **kwargs
    ):

        super().__init__(
            in_channels, num_features, growth_rate, hidden_channels,
            kernel_size,
            bands=bands, sections=sections,
            scale=scale,
            dilated=dilated, norm=norm, nonlinear=nonlinear, depth=depth,
            growth_rate_final=growth_rate_final, hidden_channels_final=hidden_channels_final,
            kernel_size_final=kernel_size_final,
            dilated_final=dilated_final, norm_final=norm_final, nonlinear_final=nonlinear_final,
            depth_final=depth_final,
            causal=causal,
            rnn_type='lstm', rnn_position=rnn_position,
            eps=eps,
            **kwargs
        )

    def get_config(self):
        config = {
            'in_channels': self.in_channels, 'num_features': self.num_features,
            'growth_rate': self.growth_rate,
            'hidden_channels': self.hidden_channels,
            'kernel_size': self.kernel_size,
            'bands': self.bands, 'sections': self.sections,
            'scale': self.scale,
            'dilated': self.dilated, 'norm': self.norm, 'nonlinear': self.nonlinear,
            'depth': self.depth,
            'growth_rate_final': self.growth_rate_final,
            'hidden_channels_final': self.hidden_channels_final,
            'kernel_size_final': self.kernel_size_final,
            'dilated_final': self.dilated_final,
            'depth_final': self.depth_final,
            'norm_final': self.norm_final, 'nonlinear_final': self.nonlinear_final,
            'causal': self.causal,
            'rnn_position': self.rnn_position,
            'eps': self.eps
        }

        return config

    @classmethod
    def build_from_config(cls, config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        in_channels = config['in_channels']
        bands = config['bands']

        sections = [
            config[band]['sections'] for band in bands
        ]
        num_features = {
            band: config[band]['num_features'] for band in bands + [FULL]
        }
        growth_rate = {
            band: config[band]['growth_rate'] for band in bands + [FULL]
        }
        hidden_channels = {
            band: config[band]['hidden_channels'] for band in bands + [FULL]
        }
        kernel_size = {
            band: config[band]['kernel_size'] for band in bands + [FULL]
        }
        scale = {
            band: config[band]['scale'] for band in bands + [FULL]
        }
        dilated = {
            band: config[band]['dilated'] for band in bands + [FULL]
        }
        norm = {
            band: config[band]['norm'] for band in bands + [FULL]
        }
        nonlinear = {
            band: config[band]['nonlinear'] for band in bands + [FULL]
        }
        depth = {
            band: config[band]['depth'] for band in bands + [FULL]
        }

        growth_rate_final, hidden_channels_final = config['final']['growth_rate'], config['final']['hidden_channels']
        kernel_size_final = config['final']['kernel_size']
        dilated_final = config['final']['dilated']
        depth_final = config['final']['depth']
        norm_final, nonlinear_final = config['final']['norm'], config['final']['nonlinear']

        causal = config['causal']
        rnn_position = config['rnn_position'] # rnn_type must be lstm

        eps = config.get('eps') or EPS

        model = cls(
            in_channels, num_features,
            growth_rate, hidden_channels,
            kernel_size,
            bands=bands, sections=sections,
            scale=scale,
            dilated=dilated, norm=norm, nonlinear=nonlinear,
            depth=depth,
            growth_rate_final=growth_rate_final, hidden_channels_final=hidden_channels_final,
            kernel_size_final=kernel_size_final,
            dilated_final=dilated_final,
            norm_final=norm_final, nonlinear_final=nonlinear_final,
            depth_final=depth_final,
            causal=causal,
            rnn_position=rnn_position,
            eps=eps
        )

        return model

    @classmethod
    def build_model(cls, model_path, load_state_dict=False):
        config = torch.load(model_path, map_location=lambda storage, loc: storage)

        in_channels, num_features = config['in_channels'], config['num_features']
        hidden_channels = config['hidden_channels']
        growth_rate = config['growth_rate']

        kernel_size = config['kernel_size']
        bands, sections = config['bands'], config['sections']
        scale = config['scale']

        dilated, norm, nonlinear = config['dilated'], config['norm'], config['nonlinear']
        depth = config['depth']

        growth_rate_final = config['growth_rate_final']
        hidden_channels_final = config['hidden_channels_final']
        kernel_size_final = config['kernel_size_final']
        dilated_final = config['dilated_final']
        depth_final = config['depth_final']
        norm_final, nonlinear_final = config['norm_final'] or True, config['nonlinear_final']

        causal = config['causal']
        rnn_position = config['rnn_position']

        eps = config.get('eps') or EPS

        model = cls(
            in_channels, num_features,
            growth_rate, hidden_channels,
            kernel_size,
            bands=bands, sections=sections,
            scale=scale,
            dilated=dilated, norm=norm, nonlinear=nonlinear,
            depth=depth,
            growth_rate_final=growth_rate_final, hidden_channels_final=hidden_channels_final,
            kernel_size_final=kernel_size_final,
            dilated_final=dilated_final,
            depth_final=depth_final,
            norm_final=norm_final, nonlinear_final=nonlinear_final,
            causal=causal,
            rnn_position=rnn_position,
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

            additional_attributes.update({
                'target': target
            })
        else:
            raise NotImplementedError("Not support task={}.".format(task))

        additional_attributes.update({
            'sample_rate': sample_rate
        })

        model_path = os.path.join(download_dir, "model", target, "{}.pth".format(model_choice))

        if not os.path.exists(model_path):
            download_pretrained_model_from_google_drive(model_id, download_dir, quiet=quiet)

        config = torch.load(model_path, map_location=lambda storage, loc: storage)
        model = cls.build_model(model_path, load_state_dict=load_state_dict)

        if task in ['musdb18', 'musdb18hq']:
            additional_attributes.update({
                'n_fft': config['n_fft'], 'hop_length': config['hop_length'],
                'window_fn': config['window_fn'],
                'sources': config['sources'],
                'n_sources': len(config['sources'])
            })

        for key, value in additional_attributes.items():
            setattr(model, key, value)

        return model

    @classmethod
    def TimeDomainWrapper(cls, base_model, n_fft, hop_length=None, window_fn='hann'):
        return MMDenseLSTMTimeDomainWrapper(base_model, n_fft, hop_length=hop_length, window_fn=window_fn)

class MMDenseLSTMTimeDomainWrapper(nn.Module):
    def __init__(self, base_model: nn.Module, n_fft, hop_length=None, window_fn='hann'):
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
            input <torch.Tensor>: (batch_size, in_channels, T)
        Returns:
            output <torch.Tensor>: (batch_size, in_channels, T)
        """
        assert input.dim() == 3, "input is expected 3D input."

        T = input.size(-1)

        mixture_spectrogram = stft(input, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window, onesided=True, return_complex=True)
        mixture_amplitude, mixture_angle = torch.abs(mixture_spectrogram), torch.angle(mixture_spectrogram)
        estimated_amplitude = self.base_model(mixture_amplitude)
        estimated_spectrogram = estimated_amplitude * torch.exp(1j * mixture_angle)
        output = istft(estimated_spectrogram, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window, onesided=True, return_complex=False, length=T)

        return output

def _test_mm_dense_lstm():
    config_path = "./data/mm_dense_lstm/parallel.yaml"
    batch_size, in_channels, n_bins, n_frames = 4, 2, 1025, 256

    input = torch.randn(batch_size, in_channels, n_bins, n_frames)
    model = MMDenseLSTM.build_from_config(config_path)

    output = model(input)

    print(model)
    print(input.size(), output.size())

def _test_mm_dense_lstm_paper():
    config_path = "./data/mm_dense_lstm/paper.yaml"
    batch_size, in_channels, n_bins, n_frames = 4, 2, 2049, 256

    input = torch.randn(batch_size, in_channels, n_bins, n_frames)
    model = MMDenseLSTM.build_from_config(config_path)

    output = model(input)

    print(model)
    print(model.num_parameters)
    print(input.size(), output.size())

if __name__ == '__main__':
    import torch

    torch.manual_seed(111)

    print('='*10, "MMDenseLSTM", '='*10)
    _test_mm_dense_lstm()
    print()

    print('='*10, "MMDenseLSTM (paper)", '='*10)
    _test_mm_dense_lstm_paper()