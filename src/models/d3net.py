import os

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.quantized as nnq
from torch.nn.modules.utils import _pair

from utils.audio import build_window
from utils.d3net import choose_layer_norm
from algorithm.frequency_mask import multichannel_wiener_filter
from transforms.stft import stft, istft
from conv import QuantizableConvTranspose2d
from models.transform import BandSplit
from models.glu import GLU2d, QuantizableGLU2d
from models.d2net import D2Block, D2BlockFixedDilation, QuantizableD2Block

"""
D3Net
    Reference: D3Net: Densely connected multidilated DenseNet for music source separation
    See https://arxiv.org/abs/2010.01733
"""

__sources__ = ['bass', 'drums', 'other', 'vocals']
FULL = 'full'
SAMPLE_RATE_MUSDB18 = 44100
EPS = 1e-12

class ParallelD3Net(nn.Module):
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
            if not isinstance(module, D3Net):
                raise ValueError("All modules must be D3Net.")

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
    def TimeDomainWrapper(cls, base_model, n_fft, hop_length=None, window_fn='hann'):
        return ParallelD3NetTimeDomainWrapper(base_model, n_fft, hop_length=hop_length, window_fn=window_fn)

    @classmethod
    def build_from_pretrained(cls, root="./pretrained", quiet=False, load_state_dict=True, **kwargs):
        import os

        from utils.utils import download_pretrained_model_from_google_drive

        task = kwargs.get('task')

        if not task in D3Net.pretrained_model_ids:
            raise KeyError("Invalid task ({}) is specified.".format(task))

        pretrained_model_ids_task = D3Net.pretrained_model_ids[task]
        additional_attributes = {}

        if task in ['musdb18', 'musdb18hq']:
            sample_rate = kwargs.get('sample_rate') or SAMPLE_RATE_MUSDB18
            config = kwargs.get('config') or "nnabla"
            sources = __sources__
            model_choice = kwargs.get('model_choice') or 'best'

            model_id = pretrained_model_ids_task[sample_rate][config]
            download_dir = os.path.join(root, D3Net.__name__, task, "sr{}".format(sample_rate), config)
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
            modules[target] = D3Net.build_model(model_path, load_state_dict=load_state_dict)

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

    @property
    def num_parameters(self):
        _num_parameters = 0

        for p in self.parameters():
            if p.requires_grad:
                _num_parameters += p.numel()

        return _num_parameters

class ParallelD3NetTimeDomainWrapper(nn.Module):
    def __init__(self, base_model: ParallelD3Net, n_fft, hop_length=None, window_fn='hann', eps=EPS):
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

class D3Net(nn.Module):
    pretrained_model_ids = {
        "musdb18": {
            SAMPLE_RATE_MUSDB18: {
                "paper": "1We9ea5qe3Hhcw28w1XZl2KKogW9wdzKF",
                "nnabla": "1B4e4e-8-T1oKzSg8WJ8RIbZ99QASamPB"
            }
        },
        "musdb18hq": {
            SAMPLE_RATE_MUSDB18: {
                "paper": "1--LWjAkX_1e4oDUkBAchu1OU1AMgt5CH",
                "nnabla": "1-5U73sNISmea_FAAaMsAjV0qjbJEaoZU"
            }
        }
    }
    def __init__(
        self,
        in_channels, num_features,
        growth_rate,
        kernel_size,
        bands=['low','middle'], sections=[256,1344],
        scale=(2,2),
        num_d2blocks=None, dilated=True, norm=True, nonlinear='relu',
        depth=None,
        growth_rate_final=None,
        kernel_size_final=None,
        dilated_final=True,
        depth_final=None,
        norm_final=True, nonlinear_final='relu',
        eps=EPS,
        **kwargs
    ):
        super().__init__()

        self.bands, self.sections = bands, sections

        self.band_split = BandSplit(sections=sections, dim=2)

        out_channels = 0
        for band in bands:
            out_channels = max(out_channels, growth_rate[band][-1])

        net = {}
        for band in bands:
            if growth_rate[band][-1] < out_channels:
                _out_channels = out_channels
            else:
                _out_channels = None
            net[band] = D3NetBackbone(in_channels, num_features[band], growth_rate[band], kernel_size[band], scale=scale[band], num_d2blocks=num_d2blocks[band], dilated=dilated[band], norm=norm[band], nonlinear=nonlinear[band], depth=depth[band], out_channels=_out_channels, eps=eps)
        net[FULL] = D3NetBackbone(in_channels, num_features[FULL], growth_rate[FULL], kernel_size[FULL], scale=scale[FULL], num_d2blocks=num_d2blocks[FULL], dilated=dilated[FULL], norm=norm[FULL], nonlinear=nonlinear[FULL], depth=depth[FULL], eps=eps)

        self.net = nn.ModuleDict(net)

        _in_channels = out_channels + growth_rate[FULL][-1] # channels for 'low' & 'middle' + channels for 'full'

        if kernel_size_final is None:
            kernel_size_final = kernel_size

        self.d2block = D2Block(_in_channels, growth_rate_final, kernel_size_final, dilated=dilated_final, depth=depth_final, norm=norm_final, nonlinear=nonlinear_final, eps=eps)
        self.norm2d = choose_layer_norm('BN', growth_rate_final, n_dims=2, eps=eps) # nn.BatchNorm2d
        self.glu2d = GLU2d(growth_rate_final, in_channels, kernel_size=(1, 1), stride=(1, 1))
        self.relu2d = nn.ReLU()

        self.scale_in, self.bias_in = nn.Parameter(torch.Tensor(sum(sections),), requires_grad=True), nn.Parameter(torch.Tensor(sum(sections),), requires_grad=True)
        self.scale_out, self.bias_out = nn.Parameter(torch.Tensor(sum(sections),), requires_grad=True), nn.Parameter(torch.Tensor(sum(sections),), requires_grad=True)

        self.in_channels, self.num_features = in_channels, num_features
        self.growth_rate = growth_rate
        self.kernel_size = kernel_size
        self.scale = scale
        self.num_d2blocks = num_d2blocks
        self.dilated, self.norm, self.nonlinear = dilated, norm, nonlinear
        self.depth = depth

        self.growth_rate_final = growth_rate_final
        self.kernel_size_final = kernel_size_final
        self.dilated_final = dilated_final
        self.depth_final = depth_final
        self.norm_final, self.nonlinear_final = norm_final, nonlinear_final

        self.eps = eps

        self._reset_parameters()

    def forward(self, input):
        """
        Args:
            input (batch_size, in_channels, n_bins, n_frames)
        Returns:
            output (batch_size, in_channels, n_bins, n_frames)
        """
        bands, sections = self.bands, self.sections
        n_bins = input.size(2)

        if sum(sections) == n_bins:
            x_valid, x_invalid = input, None
        else:
            sections = [sum(sections), n_bins - sum(sections)]
            x_valid, x_invalid = torch.split(input, sections, dim=2)

        x_valid = self.transform_affine_in(x_valid)
        x = self.band_split(x_valid)
        x_bands = []

        for band, x_band in zip(bands, x):
            x_band = self.net[band](x_band)
            x_bands.append(x_band)

        x_bands = torch.cat(x_bands, dim=2)

        x_full = self.net[FULL](x_valid)

        x = torch.cat([x_bands, x_full], dim=1)

        x = self.d2block(x)
        x = self.norm2d(x)
        x = self.glu2d(x)
        x = self.transform_affine_out(x)
        x = self.relu2d(x)

        _, _, _, n_frames = x.size()
        _, _, _, n_frames_in = input.size()
        padding_width = n_frames - n_frames_in
        padding_left = padding_width // 2
        padding_right = padding_width - padding_left

        x = F.pad(x, (-padding_left, -padding_right))

        if x_invalid is None:
            output = x
        else:
            output = torch.cat([x, x_invalid], dim=2)

        return output

    def transform_affine_in(self, input):
        eps = self.eps
        output = (input - self.bias_in.unsqueeze(dim=1)) / (torch.abs(self.scale_in.unsqueeze(dim=1)) + eps)

        return output

    def transform_affine_out(self, input):
        output = self.scale_out.unsqueeze(dim=1) * input + self.bias_out.unsqueeze(dim=1)

        return output

    def _reset_parameters(self):
        self.scale_in.data.fill_(1)
        self.bias_in.data.zero_()
        self.scale_out.data.fill_(1)
        self.bias_out.data.zero_()

    def get_config(self):
        config = {
            'in_channels': self.in_channels, 'num_features': self.num_features,
            'growth_rate': self.growth_rate,
            'kernel_size': self.kernel_size,
            'bands': self.bands, 'sections': self.sections,
            'scale': self.scale,
            'num_d2blocks': self.num_d2blocks,
            'dilated': self.dilated, 'norm': self.norm, 'nonlinear': self.nonlinear,
            'depth': self.depth,
            'growth_rate_final': self.growth_rate_final,
            'kernel_size_final': self.kernel_size_final,
            'dilated_final': self.dilated_final,
            'depth_final': self.depth_final,
            'norm_final': self.norm_final, 'nonlinear_final': self.nonlinear_final,
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
        kernel_size = {
            band: config[band]['kernel_size'] for band in bands + [FULL]
        }
        scale = {
            band: config[band]['scale'] for band in bands + [FULL]
        }
        num_d2blocks = {
            band: config[band]['num_d2blocks'] for band in bands + [FULL]
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

        growth_rate_final = config['final']['growth_rate']
        kernel_size_final = config['final']['kernel_size']
        dilated_final = config['final']['dilated']
        depth_final = config['final']['depth']
        norm_final, nonlinear_final = config['final']['norm'], config['final']['nonlinear']

        eps = config.get('eps') or EPS

        model = cls(
            in_channels, num_features,
            growth_rate,
            kernel_size,
            bands=bands, sections=sections,
            scale=scale,
            num_d2blocks=num_d2blocks, dilated=dilated, norm=norm, nonlinear=nonlinear,
            depth=depth,
            growth_rate_final=growth_rate_final,
            kernel_size_final=kernel_size_final,
            dilated_final=dilated_final,
            depth_final=depth_final,
            norm_final=norm_final, nonlinear_final=nonlinear_final,
            eps=eps
        )

        return model

    @classmethod
    def build_model(cls, model_path, load_state_dict=False):
        config = torch.load(model_path, map_location=lambda storage, loc: storage)

        in_channels, num_features = config['in_channels'], config['num_features']
        growth_rate = config['growth_rate']

        kernel_size = config['kernel_size']
        bands, sections = config['bands'], config['sections']
        scale = config['scale']

        num_d2blocks, dilated, norm, nonlinear = config['num_d2blocks'], config['dilated'], config['norm'], config['nonlinear']
        depth = config['depth']

        growth_rate_final = config['growth_rate_final']
        kernel_size_final = config['kernel_size_final']
        dilated_final = config['dilated_final']
        depth_final = config['depth_final']
        norm_final, nonlinear_final = config['norm_final'] or True, config['nonlinear_final']

        eps = config.get('eps') or EPS

        model = cls(
            in_channels, num_features,
            growth_rate,
            kernel_size,
            bands=bands, sections=sections,
            scale=scale,
            num_d2blocks=num_d2blocks, dilated=dilated, norm=norm, nonlinear=nonlinear,
            depth=depth,
            growth_rate_final=growth_rate_final,
            kernel_size_final=kernel_size_final,
            dilated_final=dilated_final,
            depth_final=depth_final,
            norm_final=norm_final, nonlinear_final=nonlinear_final,
            eps=eps
        )

        if load_state_dict:
            model.load_state_dict(config['state_dict'])

        return model

    @classmethod
    def build_from_pretrained(cls, root="./pretrained", target='vocals', quiet=False, load_state_dict=True, **kwargs):
        from utils.utils import download_pretrained_model_from_google_drive

        task = kwargs.get('task')

        if not task in cls.pretrained_model_ids:
            raise KeyError("Invalid task ({}) is specified.".format(task))

        pretrained_model_ids_task = cls.pretrained_model_ids[task]
        additional_attributes = {}

        if task in ['musdb18', 'musdb18hq']:
            sample_rate = kwargs.get('sample_rate') or SAMPLE_RATE_MUSDB18
            config = kwargs.get('config') or "nnabla"
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
        return D3NetTimeDomainWrapper(base_model, n_fft, hop_length=hop_length, window_fn=window_fn)

    @property
    def num_parameters(self):
        _num_parameters = 0

        for p in self.parameters():
            if p.requires_grad:
                _num_parameters += p.numel()

        return _num_parameters

class D3NetTimeDomainWrapper(nn.Module):
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

class D3NetBackbone(nn.Module):
    def __init__(self, in_channels, num_features, growth_rate, kernel_size, scale=(2,2), num_d2blocks=None, dilated=True, norm=True, nonlinear='relu', depth=None, out_channels=None, eps=EPS):
        """
        Args:
            in_channels <int>
            num_features <int>
            growth_rate <list<int>>: `len(growth_rate)` must be an odd number.
            kernel_size <int>
            scale <int> or <list<int>>: Upsampling and Downsampling scale
            num_d2blocks <list<int>>: # of D2 blocks
            dilated <list<bool>>
            norm <list<bool>>
            nonlinear <list<str>>
        """
        super().__init__()

        assert len(growth_rate) % 2 == 1, "`len(growth_rate)` must be an odd number."

        kernel_size = _pair(kernel_size)
        num_encoder_blocks = len(growth_rate) // 2

        # Network
        self.conv2d = nn.Conv2d(in_channels, num_features, kernel_size, stride=(1, 1))

        encoder, decoder = [], []
        encoder = Encoder(
            num_features, growth_rate[:num_encoder_blocks], kernel_size=kernel_size, down_scale=scale, num_d2blocks=num_d2blocks[:num_encoder_blocks],
            dilated=dilated[:num_encoder_blocks], norm=norm[:num_encoder_blocks], nonlinear=nonlinear[:num_encoder_blocks], depth=depth[:num_encoder_blocks],
            eps=eps
        )

        skip_channels = []
        for downsample_block in encoder.net:
            skip_channels.append(downsample_block.out_channels)        
        skip_channels = skip_channels[::-1]

        # encoder.net[-1].out_channels == skip_channels[0]
        _in_channels, _growth_rate = encoder.net[-1].out_channels, growth_rate[num_encoder_blocks]
        bottleneck_d3block = D3Block(_in_channels, _growth_rate, kernel_size=kernel_size, num_blocks=num_d2blocks[num_encoder_blocks], dilated=dilated[num_encoder_blocks], norm=norm[num_encoder_blocks], nonlinear=nonlinear[num_encoder_blocks], depth=depth[num_encoder_blocks])

        _in_channels = bottleneck_d3block.out_channels
        decoder = Decoder(
            _in_channels, skip_channels, growth_rate[num_encoder_blocks+1:], kernel_size=kernel_size, up_scale=scale, num_d2blocks=num_d2blocks[num_encoder_blocks+1:],
            dilated=dilated[num_encoder_blocks+1:], depth=depth[num_encoder_blocks+1:], norm=norm[num_encoder_blocks+1:], nonlinear=nonlinear[num_encoder_blocks+1:],
            eps=eps
        )

        self.encoder = encoder
        self.bottleneck_conv2d = bottleneck_d3block
        self.decoder = decoder

        if out_channels is not None:
            _in_channels = decoder.out_channels

            net = []
            norm2d = choose_layer_norm('BN', _in_channels, n_dims=2, eps=eps) # nn.BatchNorm2d
            net.append(norm2d)
            net.append(nn.Conv2d(_in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1)))

            self.pointwise_conv2d = nn.Sequential(*net)
        else:
            self.pointwise_conv2d = None

        self.kernel_size = kernel_size
        self.out_channels = out_channels

    def forward(self, input):
        Kh, Kw = self.kernel_size
        Ph, Pw = Kh - 1, Kw - 1
        padding_top = Ph // 2
        padding_bottom = Ph - padding_top
        padding_left = Pw // 2
        padding_right = Pw - padding_left

        input = F.pad(input, (padding_left, padding_right, padding_top, padding_bottom))

        x = self.conv2d(input)
        x, skip = self.encoder(x)
        x = self.bottleneck_conv2d(x)
        x = self.decoder(x, skip[::-1])

        if self.pointwise_conv2d:
            output = self.pointwise_conv2d(x)
        else:
            output = x

        return output

class Encoder(nn.Module):
    def __init__(self, in_channels, growth_rate, kernel_size, down_scale=(2,2), num_d2blocks=None, dilated=True, norm=True, nonlinear='relu', depth=None, eps=EPS):
        """
        Args:
            in_channels <int>: 
            growth_rate <list<int>>:
            kernel_size <tuple<int>> or <int>:
            num_d2blocks <list<int>> or <int>:
            dilated <list<bool>> or <bool>:
            norm <list<bool>> or <bool>:
            nonlinear <list<str>> or <str>:
        """
        super().__init__()

        if type(growth_rate) is list:
            num_d3blocks = len(growth_rate)
        else:
            # TODO: implement
            raise ValueError("`growth_rate` must be list.")

        if num_d2blocks is None:
            num_d2blocks = [None] * num_d3blocks
        elif type(num_d2blocks) is int:
            num_d2blocks = [num_d2blocks] * num_d3blocks
        elif type(num_d2blocks) is list:
            assert num_d3blocks == len(num_d2blocks), "Invalid length of `num_d2blocks`"
        else:
            raise ValueError("Invalid type of `num_d2blocks`.")

        if type(dilated) is bool:
            dilated = [dilated] * num_d3blocks
        elif type(dilated) is list:
            assert num_d3blocks == len(dilated), "Invalid length of `dilated`"
        else:
            raise ValueError("Invalid type of `dilated`.")

        if type(norm) is bool:
            norm = [norm] * num_d3blocks
        elif type(norm) is list:
            assert num_d3blocks == len(norm), "Invalid length of `norm`"
        else:
            raise ValueError("Invalid type of `norm`.")

        if type(nonlinear) is str:
            nonlinear = [nonlinear] * num_d3blocks
        elif type(nonlinear) is list:
            assert num_d3blocks == len(nonlinear), "Invalid length of `nonlinear`"
        else:
            raise ValueError("Invalid type of `nonlinear`.")

        if depth is None:
            depth = [None] * num_d3blocks
        elif type(depth) is int:
            depth = [depth] * num_d3blocks
        elif type(depth) is list:
            assert num_d3blocks == len(depth), "Invalid length of `depth`"
        else:
            raise ValueError("Invalid type of `depth`.")

        num_d3blocks = len(growth_rate)
        net = []

        _in_channels = in_channels

        for idx in range(num_d3blocks):
            downsample_block = DownSampleD3Block(_in_channels, growth_rate[idx], kernel_size=kernel_size, down_scale=down_scale, num_blocks=num_d2blocks[idx], dilated=dilated[idx], norm=norm[idx], nonlinear=nonlinear[idx], depth=depth[idx], eps=eps)
            net.append(downsample_block)
            _in_channels = downsample_block.out_channels

        self.net = nn.Sequential(*net)

        self.num_d3blocks = num_d3blocks

    def forward(self, input):
        num_d3blocks = self.num_d3blocks

        x = input
        skip = []

        for idx in range(num_d3blocks):
            x, x_skip = self.net[idx](x)
            skip.append(x_skip)

        output = x

        return output, skip

class Decoder(nn.Module):
    def __init__(self, in_channels, skip_channels, growth_rate, kernel_size, up_scale=(2,2), num_d2blocks=None, dilated=True, norm=True, nonlinear='relu', depth=None, eps=EPS):
        """
        Args:
            in_channels <int>: 
            skip_channels <list<int>>:
            growth_rate <list<int>>:
            kernel_size <tuple<int>> or <int>:
            num_d2blocks <list<int>> or <int>:
            dilated <list<bool>> or <bool>:
            norm <list<bool>> or <bool>:
            nonlinear <list<str>> or <str>:
        """
        super().__init__()

        if type(growth_rate) is list:
            num_d3blocks = len(growth_rate)
        else:
            # TODO: implement
            raise ValueError("`growth_rate` must be list.")

        if num_d2blocks is None:
            num_d2blocks = [None] * num_d3blocks
        elif type(num_d2blocks) is int:
            num_d2blocks = [num_d2blocks] * num_d3blocks
        elif type(num_d2blocks) is list:
            assert num_d3blocks == len(num_d2blocks), "Invalid length of `num_d2blocks`"
        else:
            raise ValueError("Invalid type of `num_d2blocks`.")

        if type(dilated) is bool:
            dilated = [dilated] * num_d3blocks
        elif type(dilated) is list:
            assert num_d3blocks == len(dilated), "Invalid length of `dilated`"
        else:
            raise ValueError("Invalid type of `dilated`.")

        if type(norm) is bool:
            norm = [norm] * num_d3blocks
        elif type(norm) is list:
            assert num_d3blocks == len(norm), "Invalid length of `norm`"
        else:
            raise ValueError("Invalid type of `norm`.")

        if type(nonlinear) is str:
            nonlinear = [nonlinear] * num_d3blocks
        elif type(nonlinear) is list:
            assert num_d3blocks == len(nonlinear), "Invalid length of `nonlinear`"
        else:
            raise ValueError("Invalid type of `nonlinear`.")

        if depth is None:
            depth = [None] * num_d3blocks
        elif type(depth) is int:
            depth = [depth] * num_d3blocks
        elif type(depth) is list:
            assert num_d3blocks == len(depth), "Invalid length of `depth`"
        else:
            raise ValueError("Invalid type of `depth`.")

        num_d3blocks = len(growth_rate)
        net = []

        _in_channels = in_channels

        for idx in range(num_d3blocks):
            upsample_block = UpSampleD3Block(_in_channels, skip_channels[idx], growth_rate[idx], kernel_size=kernel_size, up_scale=up_scale, num_blocks=num_d2blocks[idx], dilated=dilated[idx], norm=norm[idx], nonlinear=nonlinear[idx], depth=depth[idx], eps=eps)
            net.append(upsample_block)
            _in_channels = upsample_block.out_channels

        self.net = nn.Sequential(*net)

        self.num_d3blocks = num_d3blocks
        self.out_channels = _in_channels

    def forward(self, input, skip):
        num_d3blocks = self.num_d3blocks

        x = input

        for idx in range(num_d3blocks):
            x_skip = skip[idx]
            x = self.net[idx](x, x_skip)

        output = x

        return output

class DownSampleD3Block(nn.Module):
    """
    D3Block + down sample
    """
    def __init__(self, in_channels, growth_rate, kernel_size=(3,3), down_scale=(2,2), num_blocks=None, dilated=True, norm=True, nonlinear='relu', depth=None, eps=EPS):
        super().__init__()

        self.down_scale = _pair(down_scale)

        self.d3block = D3Block(in_channels, growth_rate, kernel_size, num_blocks=num_blocks, dilated=dilated, norm=norm, nonlinear=nonlinear, depth=depth, eps=eps)
        self.downsample2d = nn.AvgPool2d(kernel_size=self.down_scale, stride=self.down_scale)

        self.out_channels = self.d3block.out_channels

    def forward(self, input):
        """
        Args:
            input (batch_size, in_channels, H, W)
        Returns:
            output:
                (batch_size, growth_rate[-1], H_down, W_down) if type(growth_rate) is list<int>
                or (batch_size, growth_rate, H_down, W_down) if type(growth_rate) is int
                where H_down = H // down_scale[0] and W_down = W // down_scale[1]
            skip:
                (batch_size, growth_rate[-1], H, W) if type(growth_rate) is list<int>
                or (batch_size, growth_rate, H, W) if type(growth_rate) is int
        """
        _, _, n_bins, n_frames = input.size()

        Kh, Kw = self.down_scale
        Ph, Pw = (Kh - n_bins % Kh) % Kh, (Kw - n_frames % Kw) % Kw
        padding_top = Ph // 2
        padding_bottom = Ph - padding_top
        padding_left = Pw // 2
        padding_right = Pw - padding_left

        input = F.pad(input, (padding_left, padding_right, padding_top, padding_bottom))

        x = self.d3block(input)
        skip = x
        skip = F.pad(skip, (-padding_left, -padding_right, -padding_top, -padding_bottom))

        output = self.downsample2d(x)

        return output, skip

class UpSampleD3Block(nn.Module):
    """
    D3Block + up sample
    """
    def __init__(self, in_channels, skip_channels, growth_rate, kernel_size=(2,2), up_scale=(2,2), num_blocks=None, dilated=True, norm=True, nonlinear='relu', depth=None, eps=EPS):
        super().__init__()

        self.norm2d = choose_layer_norm('BN', in_channels, n_dims=2, eps=eps) # nn.BatchNorm2d
        self.upsample2d = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=up_scale, stride=up_scale)
        self.d3block = D3Block(in_channels + skip_channels, growth_rate, kernel_size, num_blocks=num_blocks, dilated=dilated, norm=norm, nonlinear=nonlinear, depth=depth, eps=eps)

        self.out_channels = self.d3block.out_channels

    def forward(self, input, skip):
        x = self.norm2d(input)
        x = self.upsample2d(x)

        _, _, H, W = x.size()
        _, _, H_skip, W_skip = skip.size()
        padding_height = H - H_skip
        padding_width = W - W_skip
        padding_top = padding_height // 2
        padding_bottom = padding_height - padding_top
        padding_left = padding_width // 2
        padding_right = padding_width - padding_left

        x = F.pad(x, (-padding_left, -padding_right, -padding_top, -padding_bottom))
        x = torch.cat([x, skip], dim=1)

        output = self.d3block(x)

        return output

class D3Block(nn.Module):
    def __init__(self, in_channels, growth_rate, kernel_size=(3,3), num_blocks=None, dilated=True, norm=True, nonlinear='relu', depth=None, eps=EPS):
        """
        Args:
            in_channels <int>: # of input channels
            growth_rate <int> or <list<int>>: # of output channels, TODO: <list<list<int>>>
            kernel_size <int> or <tuple<int>>: Kernel size
            num_blocks <int>: If `growth_rate` is given by list, len(growth_rate) must be equal to `num_blocks`.
            dilated <str> or <bool> or <list<bool>>: Applies dilated convolution.
            norm <bool> or <list<bool>>: Applies batch normalization.
            nonlinear <str> or <list<str>>: Applies nonlinear function.
            depth <int>: 
        """
        super().__init__()

        if type(growth_rate) is int:
            assert num_blocks is not None, "Specify `num_blocks`"
            growth_rate = [growth_rate] * num_blocks
        elif type(growth_rate) is list:
            if num_blocks is not None:
                assert num_blocks == len(growth_rate), "`num_blocks` is different from `len(growth_rate)`"
            num_blocks = len(growth_rate)
        else:
            raise ValueError("Not support growth_rate={}".format(growth_rate))

        naive_dilated = False

        if type(dilated) is str:
            if dilated == 'multi':
                pass # naive_dilated = False
            elif dilated == 'naive':
                naive_dilated = True
            else:
                raise ValueError("Not support dilated={}".format(dilated))

        if not naive_dilated:
            # w/o dilation or multi dilation
            if type(dilated) is bool:
                assert num_blocks is not None, "Specify `num_blocks`"
                dilated = [dilated] * num_blocks
            elif type(dilated) is list:
                if num_blocks is not None:
                    assert num_blocks == len(dilated), "`num_blocks` is different from `len(dilated)`"
                num_blocks = len(dilated)
            else:
                raise ValueError("Not support dilated={}".format(dilated))

        if type(norm) is bool:
            assert num_blocks is not None, "Specify `num_blocks`"
            norm = [norm] * num_blocks
        elif type(norm) is list:
            if num_blocks is not None:
                assert num_blocks == len(norm), "`num_blocks` is different from `len(norm)`"
            num_blocks = len(norm)
        else:
            raise ValueError("Not support norm={}".format(norm))

        if type(nonlinear) is str:
            assert num_blocks is not None, "Specify `num_blocks`"
            nonlinear = [nonlinear] * num_blocks
        elif type(nonlinear) is list:
            if num_blocks is not None:
                assert num_blocks == len(nonlinear), "`num_blocks` is different from `len(nonlinear)`"
            num_blocks = len(nonlinear)
        else:
            raise ValueError("Not support nonlinear={}".format(nonlinear))

        self.growth_rate = growth_rate
        self.num_blocks = num_blocks
        self.out_channels = growth_rate[-1]

        net = []

        for idx in range(num_blocks):
            if idx == 0:
                _in_channels = in_channels
            else:
                _in_channels = growth_rate[idx - 1]
            _out_channels = sum(growth_rate[idx:])

            if naive_dilated:
                dilation = 2**idx
                d2block = D2BlockFixedDilation(_in_channels, _out_channels, kernel_size=kernel_size, dilation=dilation, norm=norm[idx], nonlinear=nonlinear[idx], depth=depth, eps=eps)
            else:
                d2block = D2Block(_in_channels, _out_channels, kernel_size=kernel_size, dilated=dilated[idx], norm=norm[idx], nonlinear=nonlinear[idx], depth=depth, eps=eps)
            net.append(d2block)

        self.net = nn.Sequential(*net)

    def forward(self, input):
        """
        Args:
            input: (batch_size, in_channels, H, W)
        Returns:
            output: (batch_size, out_channels, H, W), where `out_channels` is determined by `growth_rate`.
        """
        growth_rate, num_blocks = self.growth_rate, self.num_blocks

        for idx in range(num_blocks):
            if idx == 0:
                x = input
                x_residual = 0
            else:
                _in_channels = growth_rate[idx - 1]
                sections = [_in_channels, sum(growth_rate[idx:])]
                x, x_residual = torch.split(x_residual, sections, dim=1)

            x = self.net[idx](x)
            x_residual = x_residual + x

        output = x_residual

        return output

"""
    Quantization
"""
class QuantizableD3Net(nn.Module):
    def __init__(
        self,
        in_channels, num_features,
        growth_rate,
        kernel_size,
        bands=['low','middle'], sections=[256,1344],
        scale=(2,2),
        num_d2blocks=None, dilated=True, norm=True, nonlinear='relu',
        depth=None,
        growth_rate_final=None,
        kernel_size_final=None,
        dilated_final=True,
        depth_final=None,
        norm_final=True, nonlinear_final='relu',
        eps=EPS,
        **kwargs
    ):
        super().__init__()

        self.bands, self.sections = bands, sections

        self.band_split = BandSplit(sections=sections, dim=2)
        self.affine_in = nn.Conv1d(sum(sections), sum(sections), kernel_size=1, groups=sum(sections))

        out_channels = 0
        for band in bands:
            out_channels = max(out_channels, growth_rate[band][-1])

        net = {}
        for band in bands:
            if growth_rate[band][-1] < out_channels:
                _out_channels = out_channels
            else:
                _out_channels = None
            net[band] = QuantizableD3NetBackbone(in_channels, num_features[band], growth_rate[band], kernel_size[band], scale=scale[band], num_d2blocks=num_d2blocks[band], dilated=dilated[band], norm=norm[band], nonlinear=nonlinear[band], depth=depth[band], out_channels=_out_channels, eps=eps)
        net[FULL] = QuantizableD3NetBackbone(in_channels, num_features[FULL], growth_rate[FULL], kernel_size[FULL], scale=scale[FULL], num_d2blocks=num_d2blocks[FULL], dilated=dilated[FULL], norm=norm[FULL], nonlinear=nonlinear[FULL], depth=depth[FULL], eps=eps)

        self.net = nn.ModuleDict(net)

        _in_channels = out_channels + growth_rate[FULL][-1] # channels for 'low' & 'middle' + channels for 'full'

        if kernel_size_final is None:
            kernel_size_final = kernel_size

        self.d2block = QuantizableD2Block(_in_channels, growth_rate_final, kernel_size_final, dilated=dilated_final, depth=depth_final, norm=norm_final, nonlinear=nonlinear_final, eps=eps)
        self.norm2d = choose_layer_norm('BN', growth_rate_final, n_dims=2, eps=eps) # nn.BatchNorm2d
        self.glu2d = QuantizableGLU2d(growth_rate_final, in_channels, kernel_size=(1, 1), stride=(1, 1))
        self.affine_out = nn.Conv1d(sum(sections), sum(sections), kernel_size=1, groups=sum(sections))
        self.relu2d = nn.ReLU()

        self.in_channels, self.num_features = in_channels, num_features
        self.growth_rate = growth_rate
        self.kernel_size = kernel_size
        self.scale = scale
        self.num_d2blocks = num_d2blocks
        self.dilated, self.norm, self.nonlinear = dilated, norm, nonlinear
        self.depth = depth

        self.growth_rate_final = growth_rate_final
        self.kernel_size_final = kernel_size_final
        self.dilated_final = dilated_final
        self.depth_final = depth_final
        self.norm_final, self.nonlinear_final = norm_final, nonlinear_final

        self.eps = eps

        self._reset_parameters()

    def forward(self, input):
        """
        Args:
            input (batch_size, in_channels, n_bins, n_frames)
        Returns:
            output (batch_size, in_channels, n_bins, n_frames)
        """
        bands, sections = self.bands, self.sections
        n_bins = input.size(2)

        if sum(sections) == n_bins:
            x_valid, x_invalid = input, None
        else:
            sections = [sum(sections), n_bins - sum(sections)]
            x_valid, x_invalid = torch.split(input, sections, dim=2)

        x_valid = self.transform_affine_in(x_valid)
        x = self.band_split(x_valid)
        x_bands = []

        for band, x_band in zip(bands, x):
            x_band = self.net[band](x_band)
            x_bands.append(x_band)

        x_bands = torch.cat(x_bands, dim=2)

        x_full = self.net[FULL](x_valid)

        x = torch.cat([x_bands, x_full], dim=1)

        x = self.d2block(x)
        x = self.norm2d(x)
        x = self.glu2d(x)
        x = self.transform_affine_out(x)
        x = self.relu2d(x)

        _, _, _, n_frames = x.size()
        _, _, _, n_frames_in = input.size()
        padding_width = n_frames - n_frames_in
        padding_left = padding_width // 2
        padding_right = padding_width - padding_left

        x = F.pad(x, (-padding_left, -padding_right))

        if x_invalid is None:
            output = x
        else:
            output = torch.cat([x, x_invalid], dim=2)

        return output

    def transform_affine_in(self, input):
        batch_size, in_channels, n_bins, n_frames = input.size()

        x = input.view(batch_size * in_channels, n_bins, n_frames)
        x = self.affine_in(x)
        output = x.view(batch_size, in_channels, n_bins, n_frames)

        return output

    def transform_affine_out(self, input):
        batch_size, in_channels, n_bins, n_frames = input.size()

        x = input.view(batch_size * in_channels, n_bins, n_frames)
        x = self.affine_out(x)
        output = x.view(batch_size, in_channels, n_bins, n_frames)

        return output

    def _reset_parameters(self):
        self.affine_in.weight.data.fill_(1)
        self.affine_in.bias.data.fill_(0)
        self.affine_out.weight.data.fill_(1)
        self.affine_out.bias.data.fill_(0)

    def get_config(self):
        config = {
            'in_channels': self.in_channels, 'num_features': self.num_features,
            'growth_rate': self.growth_rate,
            'kernel_size': self.kernel_size,
            'bands': self.bands, 'sections': self.sections,
            'scale': self.scale,
            'num_d2blocks': self.num_d2blocks,
            'dilated': self.dilated, 'norm': self.norm, 'nonlinear': self.nonlinear,
            'depth': self.depth,
            'growth_rate_final': self.growth_rate_final,
            'kernel_size_final': self.kernel_size_final,
            'dilated_final': self.dilated_final,
            'depth_final': self.depth_final,
            'norm_final': self.norm_final, 'nonlinear_final': self.nonlinear_final,
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
        kernel_size = {
            band: config[band]['kernel_size'] for band in bands + [FULL]
        }
        scale = {
            band: config[band]['scale'] for band in bands + [FULL]
        }
        num_d2blocks = {
            band: config[band]['num_d2blocks'] for band in bands + [FULL]
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

        growth_rate_final = config['final']['growth_rate']
        kernel_size_final = config['final']['kernel_size']
        dilated_final = config['final']['dilated']
        depth_final = config['final']['depth']
        norm_final, nonlinear_final = config['final']['norm'], config['final']['nonlinear']

        eps = config.get('eps') or EPS

        model = cls(
            in_channels, num_features,
            growth_rate,
            kernel_size,
            bands=bands, sections=sections,
            scale=scale,
            num_d2blocks=num_d2blocks, dilated=dilated, norm=norm, nonlinear=nonlinear,
            depth=depth,
            growth_rate_final=growth_rate_final,
            kernel_size_final=kernel_size_final,
            dilated_final=dilated_final,
            depth_final=depth_final,
            norm_final=norm_final, nonlinear_final=nonlinear_final,
            eps=eps
        )

        return model

    @classmethod
    def build_model(cls, model_path, load_state_dict=False):
        config = torch.load(model_path, map_location=lambda storage, loc: storage)

        in_channels, num_features = config['in_channels'], config['num_features']
        growth_rate = config['growth_rate']

        kernel_size = config['kernel_size']
        bands, sections = config['bands'], config['sections']
        scale = config['scale']

        num_d2blocks, dilated, norm, nonlinear = config['num_d2blocks'], config['dilated'], config['norm'], config['nonlinear']
        depth = config['depth']

        growth_rate_final = config['growth_rate_final']
        kernel_size_final = config['kernel_size_final']
        dilated_final = config['dilated_final']
        depth_final = config['depth_final']
        norm_final, nonlinear_final = config['norm_final'] or True, config['nonlinear_final']

        eps = config.get('eps') or EPS

        model = cls(
            in_channels, num_features,
            growth_rate,
            kernel_size,
            bands=bands, sections=sections,
            scale=scale,
            num_d2blocks=num_d2blocks, dilated=dilated, norm=norm, nonlinear=nonlinear,
            depth=depth,
            growth_rate_final=growth_rate_final,
            kernel_size_final=kernel_size_final,
            dilated_final=dilated_final,
            depth_final=depth_final,
            norm_final=norm_final, nonlinear_final=nonlinear_final,
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

class QuantizableD3NetBackbone(nn.Module):
    def __init__(self, in_channels, num_features, growth_rate, kernel_size, scale=(2,2), num_d2blocks=None, dilated=True, norm=True, nonlinear='relu', depth=None, out_channels=None, eps=EPS):
        """
        Args:
            in_channels <int>
            num_features <int>
            growth_rate <list<int>>: `len(growth_rate)` must be an odd number.
            kernel_size <int>
            scale <int> or <list<int>>: Upsampling and Downsampling scale
            num_d2blocks <list<int>>: # of D2 blocks
            dilated <list<bool>>
            norm <list<bool>>
            nonlinear <list<str>>
        """
        super().__init__()

        assert len(growth_rate) % 2 == 1, "`len(growth_rate)` must be an odd number."

        kernel_size = _pair(kernel_size)
        num_encoder_blocks = len(growth_rate) // 2

        # Network
        self.conv2d = nn.Conv2d(in_channels, num_features, kernel_size, stride=(1, 1))

        encoder, decoder = [], []
        encoder = QuantizableEncoder(
            num_features, growth_rate[:num_encoder_blocks], kernel_size=kernel_size, down_scale=scale, num_d2blocks=num_d2blocks[:num_encoder_blocks],
            dilated=dilated[:num_encoder_blocks], norm=norm[:num_encoder_blocks], nonlinear=nonlinear[:num_encoder_blocks], depth=depth[:num_encoder_blocks],
            eps=eps
        )

        skip_channels = []
        for downsample_block in encoder.net:
            skip_channels.append(downsample_block.out_channels)        
        skip_channels = skip_channels[::-1]

        # encoder.net[-1].out_channels == skip_channels[0]
        _in_channels, _growth_rate = encoder.net[-1].out_channels, growth_rate[num_encoder_blocks]
        bottleneck_d3block = QuantizableD3Block(_in_channels, _growth_rate, kernel_size=kernel_size, num_blocks=num_d2blocks[num_encoder_blocks], dilated=dilated[num_encoder_blocks], norm=norm[num_encoder_blocks], nonlinear=nonlinear[num_encoder_blocks], depth=depth[num_encoder_blocks])

        _in_channels = bottleneck_d3block.out_channels
        decoder = QuantizableDecoder(
            _in_channels, skip_channels, growth_rate[num_encoder_blocks+1:], kernel_size=kernel_size, up_scale=scale, num_d2blocks=num_d2blocks[num_encoder_blocks+1:],
            dilated=dilated[num_encoder_blocks+1:], depth=depth[num_encoder_blocks+1:], norm=norm[num_encoder_blocks+1:], nonlinear=nonlinear[num_encoder_blocks+1:],
            eps=eps
        )

        self.encoder = encoder
        self.bottleneck_conv2d = bottleneck_d3block
        self.decoder = decoder

        if out_channels is not None:
            _in_channels = decoder.out_channels

            net = []
            norm2d = choose_layer_norm('BN', _in_channels, n_dims=2, eps=eps) # nn.BatchNorm2d
            net.append(norm2d)
            net.append(nn.Conv2d(_in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1)))

            self.pointwise_conv2d = nn.Sequential(*net)
        else:
            self.pointwise_conv2d = None

        self.kernel_size = kernel_size
        self.out_channels = out_channels

    def forward(self, input):
        Kh, Kw = self.kernel_size
        Ph, Pw = Kh - 1, Kw - 1
        padding_top = Ph // 2
        padding_bottom = Ph - padding_top
        padding_left = Pw // 2
        padding_right = Pw - padding_left

        input = F.pad(input, (padding_left, padding_right, padding_top, padding_bottom))

        x = self.conv2d(input)
        x, skip = self.encoder(x)
        x = self.bottleneck_conv2d(x)
        x = self.decoder(x, skip[::-1])

        if self.pointwise_conv2d:
            output = self.pointwise_conv2d(x)
        else:
            output = x

        return output

class QuantizableEncoder(nn.Module):
    def __init__(self, in_channels, growth_rate, kernel_size, down_scale=(2,2), num_d2blocks=None, dilated=True, norm=True, nonlinear='relu', depth=None, eps=EPS):
        """
        Args:
            in_channels <int>: 
            growth_rate <list<int>>:
            kernel_size <tuple<int>> or <int>:
            num_d2blocks <list<int>> or <int>:
            dilated <list<bool>> or <bool>:
            norm <list<bool>> or <bool>:
            nonlinear <list<str>> or <str>:
        """
        super().__init__()

        if type(growth_rate) is list:
            num_d3blocks = len(growth_rate)
        else:
            # TODO: implement
            raise ValueError("`growth_rate` must be list.")

        if num_d2blocks is None:
            num_d2blocks = [None] * num_d3blocks
        elif type(num_d2blocks) is int:
            num_d2blocks = [num_d2blocks] * num_d3blocks
        elif type(num_d2blocks) is list:
            assert num_d3blocks == len(num_d2blocks), "Invalid length of `num_d2blocks`"
        else:
            raise ValueError("Invalid type of `num_d2blocks`.")

        if type(dilated) is bool:
            dilated = [dilated] * num_d3blocks
        elif type(dilated) is list:
            assert num_d3blocks == len(dilated), "Invalid length of `dilated`"
        else:
            raise ValueError("Invalid type of `dilated`.")

        if type(norm) is bool:
            norm = [norm] * num_d3blocks
        elif type(norm) is list:
            assert num_d3blocks == len(norm), "Invalid length of `norm`"
        else:
            raise ValueError("Invalid type of `norm`.")

        if type(nonlinear) is str:
            nonlinear = [nonlinear] * num_d3blocks
        elif type(nonlinear) is list:
            assert num_d3blocks == len(nonlinear), "Invalid length of `nonlinear`"
        else:
            raise ValueError("Invalid type of `nonlinear`.")

        if depth is None:
            depth = [None] * num_d3blocks
        elif type(depth) is int:
            depth = [depth] * num_d3blocks
        elif type(depth) is list:
            assert num_d3blocks == len(depth), "Invalid length of `depth`"
        else:
            raise ValueError("Invalid type of `depth`.")

        num_d3blocks = len(growth_rate)
        net = []

        _in_channels = in_channels

        for idx in range(num_d3blocks):
            downsample_block = QuantizableDownSampleD3Block(_in_channels, growth_rate[idx], kernel_size=kernel_size, down_scale=down_scale, num_blocks=num_d2blocks[idx], dilated=dilated[idx], norm=norm[idx], nonlinear=nonlinear[idx], depth=depth[idx], eps=eps)
            net.append(downsample_block)
            _in_channels = downsample_block.out_channels

        self.net = nn.Sequential(*net)

        self.num_d3blocks = num_d3blocks

    def forward(self, input):
        num_d3blocks = self.num_d3blocks

        x = input
        skip = []

        for idx in range(num_d3blocks):
            x, x_skip = self.net[idx](x)
            skip.append(x_skip)

        output = x

        return output, skip

class QuantizableDecoder(nn.Module):
    def __init__(self, in_channels, skip_channels, growth_rate, kernel_size, up_scale=(2,2), num_d2blocks=None, dilated=True, norm=True, nonlinear='relu', depth=None, eps=EPS):
        """
        Args:
            in_channels <int>: 
            skip_channels <list<int>>:
            growth_rate <list<int>>:
            kernel_size <tuple<int>> or <int>:
            num_d2blocks <list<int>> or <int>:
            dilated <list<bool>> or <bool>:
            norm <list<bool>> or <bool>:
            nonlinear <list<str>> or <str>:
        """
        super().__init__()

        if type(growth_rate) is list:
            num_d3blocks = len(growth_rate)
        else:
            # TODO: implement
            raise ValueError("`growth_rate` must be list.")

        if num_d2blocks is None:
            num_d2blocks = [None] * num_d3blocks
        elif type(num_d2blocks) is int:
            num_d2blocks = [num_d2blocks] * num_d3blocks
        elif type(num_d2blocks) is list:
            assert num_d3blocks == len(num_d2blocks), "Invalid length of `num_d2blocks`"
        else:
            raise ValueError("Invalid type of `num_d2blocks`.")

        if type(dilated) is bool:
            dilated = [dilated] * num_d3blocks
        elif type(dilated) is list:
            assert num_d3blocks == len(dilated), "Invalid length of `dilated`"
        else:
            raise ValueError("Invalid type of `dilated`.")

        if type(norm) is bool:
            norm = [norm] * num_d3blocks
        elif type(norm) is list:
            assert num_d3blocks == len(norm), "Invalid length of `norm`"
        else:
            raise ValueError("Invalid type of `norm`.")

        if type(nonlinear) is str:
            nonlinear = [nonlinear] * num_d3blocks
        elif type(nonlinear) is list:
            assert num_d3blocks == len(nonlinear), "Invalid length of `nonlinear`"
        else:
            raise ValueError("Invalid type of `nonlinear`.")

        if depth is None:
            depth = [None] * num_d3blocks
        elif type(depth) is int:
            depth = [depth] * num_d3blocks
        elif type(depth) is list:
            assert num_d3blocks == len(depth), "Invalid length of `depth`"
        else:
            raise ValueError("Invalid type of `depth`.")

        num_d3blocks = len(growth_rate)
        net = []

        _in_channels = in_channels

        for idx in range(num_d3blocks):
            upsample_block = QuantizableUpSampleD3Block(_in_channels, skip_channels[idx], growth_rate[idx], kernel_size=kernel_size, up_scale=up_scale, num_blocks=num_d2blocks[idx], dilated=dilated[idx], norm=norm[idx], nonlinear=nonlinear[idx], depth=depth[idx], eps=eps)
            net.append(upsample_block)
            _in_channels = upsample_block.out_channels

        self.net = nn.Sequential(*net)

        self.num_d3blocks = num_d3blocks
        self.out_channels = _in_channels

    def forward(self, input, skip):
        num_d3blocks = self.num_d3blocks

        x = input

        for idx in range(num_d3blocks):
            x_skip = skip[idx]
            x = self.net[idx](x, x_skip)

        output = x

        return output

class QuantizableDownSampleD3Block(nn.Module):
    """
    D3Block + down sample
    """
    def __init__(self, in_channels, growth_rate, kernel_size=(3,3), down_scale=(2,2), num_blocks=None, dilated=True, norm=True, nonlinear='relu', depth=None, eps=EPS):
        super().__init__()

        self.down_scale = _pair(down_scale)

        self.d3block = QuantizableD3Block(in_channels, growth_rate, kernel_size, num_blocks=num_blocks, dilated=dilated, norm=norm, nonlinear=nonlinear, depth=depth, eps=eps)
        self.downsample2d = nn.AvgPool2d(kernel_size=self.down_scale, stride=self.down_scale)

        self.out_channels = self.d3block.out_channels

    def forward(self, input):
        """
        Args:
            input (batch_size, in_channels, H, W)
        Returns:
            output:
                (batch_size, growth_rate[-1], H_down, W_down) if type(growth_rate) is list<int>
                or (batch_size, growth_rate, H_down, W_down) if type(growth_rate) is int
                where H_down = H // down_scale[0] and W_down = W // down_scale[1]
            skip:
                (batch_size, growth_rate[-1], H, W) if type(growth_rate) is list<int>
                or (batch_size, growth_rate, H, W) if type(growth_rate) is int
        """
        _, _, n_bins, n_frames = input.size()

        Kh, Kw = self.down_scale
        Ph, Pw = (Kh - n_bins % Kh) % Kh, (Kw - n_frames % Kw) % Kw
        padding_top = Ph // 2
        padding_bottom = Ph - padding_top
        padding_left = Pw // 2
        padding_right = Pw - padding_left

        input = F.pad(input, (padding_left, padding_right, padding_top, padding_bottom))

        x = self.d3block(input)
        skip = x
        skip = F.pad(skip, (-padding_left, -padding_right, -padding_top, -padding_bottom))

        output = self.downsample2d(x)

        return output, skip

class QuantizableUpSampleD3Block(nn.Module):
    """
    D3Block + up sample
    """
    def __init__(self, in_channels, skip_channels, growth_rate, kernel_size=(2,2), up_scale=(2,2), num_blocks=None, dilated=True, norm=True, nonlinear='relu', depth=None, eps=EPS):
        super().__init__()

        self.norm2d = choose_layer_norm('BN', in_channels, n_dims=2, eps=eps) # nn.BatchNorm2d
        self.upsample2d = QuantizableConvTranspose2d(in_channels, in_channels, kernel_size=up_scale, stride=up_scale)
        self.d3block = QuantizableD3Block(in_channels + skip_channels, growth_rate, kernel_size, num_blocks=num_blocks, dilated=dilated, norm=norm, nonlinear=nonlinear, depth=depth, eps=eps)

        self.out_channels = self.d3block.out_channels

    def forward(self, input, skip):
        x = self.norm2d(input)
        x = self.upsample2d(x)

        _, _, H, W = x.size()
        _, _, H_skip, W_skip = skip.size()
        padding_height = H - H_skip
        padding_width = W - W_skip
        padding_top = padding_height // 2
        padding_bottom = padding_height - padding_top
        padding_left = padding_width // 2
        padding_right = padding_width - padding_left

        x = F.pad(x, (-padding_left, -padding_right, -padding_top, -padding_bottom))
        x = torch.cat([x, skip], dim=1)

        output = self.d3block(x)

        return output

class QuantizableD3Block(nn.Module):
    def __init__(self, in_channels, growth_rate, kernel_size=(3,3), num_blocks=None, dilated=True, norm=True, nonlinear='relu', depth=None, eps=EPS):
        """
        Args:
            in_channels <int>: # of input channels
            growth_rate <int> or <list<int>>: # of output channels, TODO: <list<list<int>>>
            kernel_size <int> or <tuple<int>>: Kernel size
            num_blocks <int>: If `growth_rate` is given by list, len(growth_rate) must be equal to `num_blocks`.
            dilated <str> or <bool> or <list<bool>>: Applies dilated convolution.
            norm <bool> or <list<bool>>: Applies batch normalization.
            nonlinear <str> or <list<str>>: Applies nonlinear function.
            depth <int>: 
        """
        super().__init__()

        if type(growth_rate) is int:
            assert num_blocks is not None, "Specify `num_blocks`"
            growth_rate = [growth_rate] * num_blocks
        elif type(growth_rate) is list:
            if num_blocks is not None:
                assert num_blocks == len(growth_rate), "`num_blocks` is different from `len(growth_rate)`"
            num_blocks = len(growth_rate)
        else:
            raise ValueError("Not support growth_rate={}".format(growth_rate))

        naive_dilated = False

        if type(dilated) is str:
            if dilated == 'multi':
                pass # naive_dilated = False
            elif dilated == 'naive':
                naive_dilated = True
            else:
                raise ValueError("Not support dilated={}".format(dilated))

        if not naive_dilated:
            # w/o dilation or multi dilation
            if type(dilated) is bool:
                assert num_blocks is not None, "Specify `num_blocks`"
                dilated = [dilated] * num_blocks
            elif type(dilated) is list:
                if num_blocks is not None:
                    assert num_blocks == len(dilated), "`num_blocks` is different from `len(dilated)`"
                num_blocks = len(dilated)
            else:
                raise ValueError("Not support dilated={}".format(dilated))

        if type(norm) is bool:
            assert num_blocks is not None, "Specify `num_blocks`"
            norm = [norm] * num_blocks
        elif type(norm) is list:
            if num_blocks is not None:
                assert num_blocks == len(norm), "`num_blocks` is different from `len(norm)`"
            num_blocks = len(norm)
        else:
            raise ValueError("Not support norm={}".format(norm))

        if type(nonlinear) is str:
            assert num_blocks is not None, "Specify `num_blocks`"
            nonlinear = [nonlinear] * num_blocks
        elif type(nonlinear) is list:
            if num_blocks is not None:
                assert num_blocks == len(nonlinear), "`num_blocks` is different from `len(nonlinear)`"
            num_blocks = len(nonlinear)
        else:
            raise ValueError("Not support nonlinear={}".format(nonlinear))

        self.growth_rate = growth_rate
        self.num_blocks = num_blocks
        self.out_channels = growth_rate[-1]
        self.float_ops = nnq.FloatFunctional()

        net = []

        for idx in range(num_blocks):
            if idx == 0:
                _in_channels = in_channels
            else:
                _in_channels = growth_rate[idx - 1]
            _out_channels = sum(growth_rate[idx:])

            if naive_dilated:
                raise NotImplementedError("Not support naive_dilated=True.")
            else:
                d2block = QuantizableD2Block(_in_channels, _out_channels, kernel_size=kernel_size, dilated=dilated[idx], norm=norm[idx], nonlinear=nonlinear[idx], depth=depth, eps=eps)
            net.append(d2block)

        self.net = nn.Sequential(*net)

    def forward(self, input):
        """
        Args:
            input: (batch_size, in_channels, H, W)
        Returns:
            output: (batch_size, out_channels, H, W), where `out_channels` is determined by `growth_rate`.
        """
        growth_rate, num_blocks = self.growth_rate, self.num_blocks

        for idx in range(num_blocks):
            if idx == 0:
                x = input
                x_residual = None
            else:
                _in_channels = growth_rate[idx - 1]
                sections = [_in_channels, sum(growth_rate[idx:])]
                x, x_residual = torch.split(x_residual, sections, dim=1)

            x = self.net[idx](x)

            if x_residual is None:
                x_residual = x
            else:
                x_residual = self.float_ops.add(x_residual, x)

        output = x_residual

        return output

def _test_d3block():
    batch_size = 4
    n_bins, n_frames = 16, 64
    in_channels = 3
    growth_rate = 2
    kernel_size = (3, 3)
    num_blocks, depth = 2, 4

    input = torch.randn(batch_size, in_channels, n_bins, n_frames)
    model = D3Block(in_channels, growth_rate, kernel_size=kernel_size, num_blocks=num_blocks, depth=depth)

    print(model)
    output = model(input)
    print(input.size(), output.size(), model.out_channels)
    print()

    growth_rate = [3, 4, 5, 6]
    model = D3Block(in_channels, growth_rate, kernel_size=kernel_size, depth=depth)

    print(model)
    output = model(input)
    print(input.size(), output.size(), model.out_channels)

def _test_down_d3block():
    batch_size = 4
    n_bins, n_frames = 16, 64
    in_channels = 3
    growth_rate = 2
    kernel_size = (3, 3)
    down_scale = (2, 2)
    num_blocks, depth = 2, 4

    input = torch.randn(batch_size, in_channels, n_bins, n_frames)
    model = DownSampleD3Block(in_channels, growth_rate, kernel_size=kernel_size, down_scale=down_scale, num_blocks=num_blocks, depth=depth)

    print(model)
    output, skip = model(input)
    print(input.size(), output.size(), skip.size(), model.out_channels)
    print()

    growth_rate = [3, 4, 5, 6]
    model = DownSampleD3Block(in_channels, growth_rate, kernel_size=kernel_size, down_scale=down_scale, depth=depth)

    print(model)
    output, skip = model(input)
    print(input.size(), output.size(), skip.size(), model.out_channels)

def _test_encoder():
    batch_size = 4
    n_bins, n_frames = 16, 64
    in_channels = 32

    growth_rate = [2, 3, 4]
    kernel_size = 3
    num_d2blocks = 2
    
    depth = [2, 2, 3]
    input = torch.randn(batch_size, in_channels, n_bins, n_frames)
    encoder = Encoder(in_channels, growth_rate, kernel_size, num_d2blocks=num_d2blocks, depth=depth)
    output, skip = encoder(input)

    print(encoder)
    print(input.size(), output.size())
    for _skip in skip:
        print(_skip.size())
    print()

    depth = 2
    encoder = Encoder(in_channels, growth_rate, kernel_size, num_d2blocks=num_d2blocks, depth=depth)
    output, skip = encoder(input)

    print(encoder)
    print(input.size(), output.size())

    for _skip in skip:
        print(_skip.size())
    print()

def _test_d3net_backbone():
    batch_size = 4
    n_bins, n_frames = 16, 64
    in_channels, num_features = 2, 32

    growth_rate = [2, 3, 4, 3, 2]
    kernel_size = 3
    num_d2blocks = [2, 2, 2, 2, 2]

    dilated = [True, True, True, True, True]
    norm = [True, True, True, True, True]
    nonlinear = ['relu', 'relu', 'relu', 'relu', 'relu']
    depth = [3, 3, 4, 2, 2]
    input = torch.randn(batch_size, in_channels, n_bins, n_frames)

    model = D3NetBackbone(in_channels, num_features, growth_rate, kernel_size, num_d2blocks=num_d2blocks, dilated=dilated, norm=norm, nonlinear=nonlinear, depth=depth)

    print(model)

    output = model(input)

    print(input.size(), output.size())

def _test_d3net_wo_dilation():
    config_path = "./data/d3net/wo_dilation.yaml"
    batch_size, in_channels, n_bins, n_frames = 4, 2, 257, 140 # 4, 2, 2049, 256

    input = torch.randn(batch_size, in_channels, n_bins, n_frames)
    model = D3Net.build_from_config(config_path)

    output = model(input)

    print(model)
    print(input.size(), output.size())

def _test_d3net_naive_dilation():
    config_path = "./data/d3net/naive_dilation.yaml"
    batch_size, in_channels, n_bins, n_frames = 4, 2, 257, 140 # 4, 2, 2049, 256

    input = torch.randn(batch_size, in_channels, n_bins, n_frames)
    model = D3Net.build_from_config(config_path)

    output = model(input)

    print(model)
    print(input.size(), output.size())

def _test_d3net():
    config_path = "./data/d3net/paper.yaml"
    batch_size, in_channels, n_bins, n_frames = 4, 2, 257, 140 # 4, 2, 2049, 256

    input = torch.randn(batch_size, in_channels, n_bins, n_frames)
    model = D3Net.build_from_config(config_path)

    output = model(input)

    print(model)
    print(input.size(), output.size())

if __name__ == '__main__':
    torch.manual_seed(111)

    print('='*10, "D3Block", '='*10)
    _test_d3block()
    print()

    print('='*10, "DownSampleD3Block", '='*10)
    _test_down_d3block()
    print()

    print('='*10, "Encoder", '='*10)
    _test_encoder()
    print()

    print('='*10, "D3Net backbone", '='*10)
    _test_d3net_backbone()
    print()

    print('='*10, "D3Net (w/o dilation)", '='*10)
    _test_d3net_wo_dilation()
    print()

    print('='*10, "D3Net (naive dilation)", '='*10)
    _test_d3net_naive_dilation()
    print()

    print('='*10, "D3Net", '='*10)
    _test_d3net()