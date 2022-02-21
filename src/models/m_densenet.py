import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.quantized as nnq
from torch.nn.modules.utils import _pair

from utils.audio import build_window
from utils.m_densenet import choose_layer_norm, choose_nonlinear
from transforms.stft import stft, istft
from models.glu import GLU2d

"""
Reference: Multi-scale Multi-band DenseNets for Audio Source Separation
See https://arxiv.org/abs/1706.09588
"""

EPS = 1e-12

class ParallelMDenseNet(nn.Module):
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
            if not isinstance(module, MDenseNet):
                raise ValueError("All modules must be MDenseNet.")

            if in_channels is None:
                in_channels = module.in_channels
            else:
                assert in_channels == module.in_channels, "`in_channels` are different among modules."

        self.net = modules

        self.in_channels = in_channels
        self.sources = sources

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

class MDenseNet(nn.Module):
    """
    Multi-scale DenseNet
    """
    def __init__(
        self,
        in_channels, num_features,
        growth_rate,
        kernel_size,
        max_bin=1367,
        scale=(2,2),
        dilated=False, norm=True, nonlinear='relu',
        depth=None,
        growth_rate_final=None,
        kernel_size_final=None,
        dilated_final=False,
        norm_final=True, nonlinear_final='relu',
        depth_final=None,
        eps=EPS,
        **kwargs
    ):
        super().__init__()

        self.net = MDenseNetBackbone(in_channels, num_features, growth_rate, kernel_size, scale=scale, dilated=dilated, norm=norm, nonlinear=nonlinear, depth=depth, eps=eps)

        self.relu2d = nn.ReLU()

        _in_channels = growth_rate[-1] # output channels of self.net
        self.dense_block = DenseBlock(_in_channels, growth_rate_final, kernel_size_final, dilated=dilated_final, depth=depth_final, norm=norm_final, nonlinear=nonlinear_final, eps=eps)
        self.norm2d = choose_layer_norm('BN', growth_rate_final, n_dims=2, eps=eps) # nn.BatchNorm2d
        self.glu2d = GLU2d(growth_rate_final, in_channels, kernel_size=(1,1), stride=(1,1))
        self.relu2d = nn.ReLU()

        self.scale_in, self.bias_in = nn.Parameter(torch.Tensor(max_bin,), requires_grad=True), nn.Parameter(torch.Tensor(max_bin,), requires_grad=True)
        self.scale_out, self.bias_out = nn.Parameter(torch.Tensor(max_bin,), requires_grad=True), nn.Parameter(torch.Tensor(max_bin,), requires_grad=True)

        self.max_bin = max_bin
        self.in_channels, self.num_features = in_channels, num_features
        self.growth_rate = growth_rate
        self.kernel_size = kernel_size
        self.scale = scale
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
        max_bin = self.max_bin
        n_bins = input.size(2)

        if max_bin == n_bins:
            x_valid, x_invalid = input, None
        else:
            sections = [max_bin, n_bins - max_bin]
            x_valid, x_invalid = torch.split(input, sections, dim=2)

        x = self.transform_affine_in(x_valid)
        x = self.net(x)
        x = self.dense_block(x)
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
            'max_bin': self.max_bin,
            'scale': self.scale,
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

        max_bin = config['max_bin']
        num_features = config['num_features']
        growth_rate = config['growth_rate']
        kernel_size = config['kernel_size']
        scale = config['scale']
        dilated = config['dilated']
        norm = config['norm']
        nonlinear = config['nonlinear']
        depth = config['depth']

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
            max_bin=max_bin,
            scale=scale,
            dilated=dilated, norm=norm, nonlinear=nonlinear,
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
        max_bin = config['max_bin']
        scale = config['scale']

        dilated, norm, nonlinear = config['dilated'], config['norm'], config['nonlinear']
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
            max_bin=max_bin,
            scale=scale,
            dilated=dilated, norm=norm, nonlinear=nonlinear,
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
    def TimeDomainWrapper(cls, base_model, n_fft, hop_length=None, window_fn='hann'):
        return MDenseNetTimeDomainWrapper(base_model, n_fft, hop_length=hop_length, window_fn=window_fn)
    
    @property
    def num_parameters(self):
        _num_parameters = 0
        
        for p in self.parameters():
            if p.requires_grad:
                _num_parameters += p.numel()
                
        return _num_parameters

class MDenseNetTimeDomainWrapper(nn.Module):
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

class MDenseNetBackbone(nn.Module):
    def __init__(self, in_channels, num_features, growth_rate, kernel_size, scale=(2,2), dilated=False, norm=True, nonlinear='relu', depth=None, out_channels=None, eps=EPS):
        """
        Args:
            in_channels <int>
            num_features <int>
            growth_rate <list<int>>: `len(growth_rate)` must be an odd number.
            kernel_size <int> or <tuple<int>>
            scale <int> or <list<int>>: Upsampling and Downsampling scale
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
            num_features, growth_rate[:num_encoder_blocks], kernel_size=kernel_size, down_scale=scale,
            dilated=dilated[:num_encoder_blocks], norm=norm[:num_encoder_blocks], nonlinear=nonlinear[:num_encoder_blocks], depth=depth[:num_encoder_blocks],
            eps=eps
        )

        _in_channels, _growth_rate = growth_rate[num_encoder_blocks - 1], growth_rate[num_encoder_blocks]

        bottleneck_dense_block = DenseBlock(
            _in_channels, _growth_rate,
            kernel_size=kernel_size,
            dilated=dilated[num_encoder_blocks], norm=norm[num_encoder_blocks], nonlinear=nonlinear[num_encoder_blocks], depth=depth[num_encoder_blocks]
        )

        _in_channels = _growth_rate
        skip_channels = growth_rate[num_encoder_blocks - 1::-1]

        decoder = Decoder(
            _in_channels, skip_channels, growth_rate[num_encoder_blocks+1:], kernel_size=kernel_size, up_scale=scale,
            dilated=dilated[num_encoder_blocks+1:], depth=depth[num_encoder_blocks+1:], norm=norm[num_encoder_blocks+1:], nonlinear=nonlinear[num_encoder_blocks+1:],
            eps=eps
        )

        self.encoder = encoder
        self.bottleneck_conv2d = bottleneck_dense_block
        self.decoder = decoder

        if out_channels is not None:
            _in_channels = growth_rate[-1]

            net = []
            norm2d = choose_layer_norm('BN', _in_channels, n_dims=2, eps=eps) # nn.BatchNorm2d
            net.append(norm2d)
            net.append(nn.Conv2d(_in_channels, out_channels, kernel_size=(1,1), stride=(1,1)))

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
    def __init__(self, in_channels, growth_rate, kernel_size, down_scale=(2,2), dilated=False, norm=True, nonlinear='relu', depth=None, eps=EPS):
        """
        Args:
            in_channels <int>: 
            growth_rate <list<int>>:
            kernel_size <tuple<int>> or <int>:
            dilated <list<bool>> or <bool>:
            norm <list<bool>> or <bool>:
            nonlinear <list<str>> or <str>:
            depth <list<int>> or <int>:
        """
        super().__init__()

        if type(growth_rate) is list:
            num_dense_blocks = len(growth_rate)
        else:
            # TODO: implement
            raise ValueError("`growth_rate` must be list.")

        if type(dilated) is bool:
            dilated = [dilated] * num_dense_blocks
        elif type(dilated) is list:
            assert num_dense_blocks == len(dilated), "Invalid length of `dilated`"
        else:
            raise ValueError("Invalid type of `dilated`.")

        if type(norm) is bool:
            norm = [norm] * num_dense_blocks
        elif type(norm) is list:
            assert num_dense_blocks == len(norm), "Invalid length of `norm`"
        else:
            raise ValueError("Invalid type of `norm`.")

        if type(nonlinear) is str:
            nonlinear = [nonlinear] * num_dense_blocks
        elif type(nonlinear) is list:
            assert num_dense_blocks == len(nonlinear), "Invalid length of `nonlinear`"
        else:
            raise ValueError("Invalid type of `nonlinear`.")

        if depth is None:
            depth = [None] * num_dense_blocks
        elif type(depth) is int:
            depth = [depth] * num_dense_blocks
        elif type(depth) is list:
            assert num_dense_blocks == len(depth), "Invalid length of `depth`"
        else:
            raise ValueError("Invalid type of `depth`.")

        num_dense_blocks = len(growth_rate)
        net = []

        _in_channels = in_channels

        for idx in range(num_dense_blocks):
            downsample_block = DownSampleDenseBlock(_in_channels, growth_rate[idx], kernel_size=kernel_size, down_scale=down_scale, dilated=dilated[idx], norm=norm[idx], nonlinear=nonlinear[idx], depth=depth[idx], eps=eps)
            net.append(downsample_block)
            _in_channels = growth_rate[idx]

        self.net = nn.Sequential(*net)

        self.num_dense_blocks = num_dense_blocks

    def forward(self, input):
        num_dense_blocks = self.num_dense_blocks

        x = input
        skip = []

        for idx in range(num_dense_blocks):
            x, x_skip = self.net[idx](x)
            skip.append(x_skip)

        output = x

        return output, skip

class Decoder(nn.Module):
    def __init__(self, in_channels, skip_channels, growth_rate, kernel_size, up_scale=(2,2), dilated=False, norm=True, nonlinear='relu', depth=None, eps=EPS):
        """
        Args:
            in_channels <int>: 
            skip_channels <list<int>>:
            growth_rate <list<int>>:
            kernel_size <tuple<int>> or <int>:
            dilated <list<bool>> or <bool>:
            norm <list<bool>> or <bool>:
            nonlinear <list<str>> or <str>:
            depth <list<int>> or <int>:
        """
        super().__init__()

        if type(growth_rate) is list:
            num_dense_blocks = len(growth_rate)
        else:
            # TODO: implement
            raise ValueError("`growth_rate` must be list.")

        if type(dilated) is bool:
            dilated = [dilated] * num_dense_blocks
        elif type(dilated) is list:
            assert num_dense_blocks == len(dilated), "Invalid length of `dilated`"
        else:
            raise ValueError("Invalid type of `dilated`.")

        if type(norm) is bool:
            norm = [norm] * num_dense_blocks
        elif type(norm) is list:
            assert num_dense_blocks == len(norm), "Invalid length of `norm`"
        else:
            raise ValueError("Invalid type of `norm`.")

        if type(nonlinear) is str:
            nonlinear = [nonlinear] * num_dense_blocks
        elif type(nonlinear) is list:
            assert num_dense_blocks == len(nonlinear), "Invalid length of `nonlinear`"
        else:
            raise ValueError("Invalid type of `nonlinear`.")

        if depth is None:
            depth = [None] * num_dense_blocks
        elif type(depth) is int:
            depth = [depth] * num_dense_blocks
        elif type(depth) is list:
            assert num_dense_blocks == len(depth), "Invalid length of `depth`"
        else:
            raise ValueError("Invalid type of `depth`.")

        num_dense_blocks = len(growth_rate)
        net = []

        _in_channels = in_channels

        for idx in range(num_dense_blocks):
            upsample_block = UpSampleDenseBlock(_in_channels, skip_channels[idx], growth_rate[idx], kernel_size=kernel_size, up_scale=up_scale, dilated=dilated[idx], norm=norm[idx], nonlinear=nonlinear[idx], depth=depth[idx], eps=eps)
            net.append(upsample_block)
            _in_channels = growth_rate[idx]

        self.net = nn.Sequential(*net)

        self.num_dense_blocks = num_dense_blocks

    def forward(self, input, skip):
        num_dense_blocks = self.num_dense_blocks

        x = input

        for idx in range(num_dense_blocks):
            x_skip = skip[idx]
            x = self.net[idx](x, x_skip)

        output = x

        return output

class DownSampleDenseBlock(nn.Module):
    """
    DenseBlock + down sample
    """
    def __init__(self, in_channels, growth_rate, kernel_size=(3,3), down_scale=(2,2), dilated=False, norm=True, nonlinear='relu', depth=None, eps=EPS):
        super().__init__()

        self.down_scale = _pair(down_scale)

        self.dense_block = DenseBlock(in_channels, growth_rate, kernel_size, dilated=dilated, norm=norm, nonlinear=nonlinear, depth=depth, eps=eps)
        self.downsample2d = nn.AvgPool2d(kernel_size=self.down_scale, stride=self.down_scale)
        self.out_channels = self.dense_block.out_channels

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

        x = self.dense_block(input)
        skip = x
        skip = F.pad(skip, (-padding_left, -padding_right, -padding_top, -padding_bottom))

        output = self.downsample2d(x)

        return output, skip

class UpSampleDenseBlock(nn.Module):
    """
    DenseBlock + up sample
    """
    def __init__(self, in_channels, skip_channels, growth_rate, kernel_size=(2,2), up_scale=(2,2), dilated=False, norm=True, nonlinear='relu', depth=None, eps=EPS):
        super().__init__()

        self.norm2d = choose_layer_norm('BN', in_channels, n_dims=2, eps=eps) # nn.BatchNorm2d
        self.upsample2d = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=up_scale, stride=up_scale)
        self.dense_block = DenseBlock(in_channels + skip_channels, growth_rate, kernel_size, dilated=dilated, norm=norm, nonlinear=nonlinear, depth=depth, eps=eps)
        self.out_channels = self.dense_block.out_channels

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

        output = self.dense_block(x)

        return output

class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, kernel_size, depth=None, dilated=False, norm=True, nonlinear='relu', eps=EPS):
        """
        Args:
            in_channels <int>: # of input channels
            growth_rate <int> or <list<int>>: # of output channels
            kernel_size <int> or <tuple<int>>: Kernel size
            dilated <bool> or <list<bool>>: Applies dilated convolution.
            norm <bool> or <list<bool>>: Applies batch normalization.
            nonlinear <str> or <list<str>>: Applies nonlinear function.
            depth <int>: If `growth_rate` is given by list, len(growth_rate) must be equal to `depth`.
        """
        super().__init__()

        if type(growth_rate) is int:
            assert depth is not None, "Specify `depth`"
            growth_rate = [growth_rate] * depth
        elif type(growth_rate) is list:
            if depth is not None:
                assert depth == len(growth_rate), "`depth` is different from `len(growth_rate)`"
            depth = len(growth_rate)
        else:
            raise ValueError("Not support growth_rate={}".format(growth_rate))

        if type(dilated) is bool:
            assert depth is not None, "Specify `depth`"
            dilated = [dilated] * depth
        elif type(dilated) is list:
            if depth is not None:
                assert depth == len(dilated), "`depth` is different from `len(dilated)`"
            depth = len(dilated)
        else:
            raise ValueError("Not support dilated={}".format(dilated))

        if type(norm) is bool:
            assert depth is not None, "Specify `depth`"
            norm = [norm] * depth
        elif type(norm) is list:
            if depth is not None:
                assert depth == len(norm), "`depth` is different from `len(norm)`"
            depth = len(norm)
        else:
            raise ValueError("Not support norm={}".format(norm))

        if type(nonlinear) is bool or type(nonlinear) is str:
            assert depth is not None, "Specify `depth`"
            nonlinear = [nonlinear] * depth
        elif type(nonlinear) is list:
            if depth is not None:
                assert depth == len(nonlinear), "`depth` is different from `len(nonlinear)`"
            depth = len(nonlinear)
        else:
            raise ValueError("Not support nonlinear={}".format(nonlinear))

        self.growth_rate = growth_rate
        self.depth = depth

        net = []

        for idx in range(depth):
            if idx == 0:
                _in_channels = in_channels
            else:
                _in_channels = growth_rate[idx - 1]

            _out_channels = sum(growth_rate[idx:])

            if dilated[idx]:
                dilation = 2**idx
            else:
                dilation = 1

            conv_block = ConvBlock2d(_in_channels, _out_channels, kernel_size=kernel_size, stride=1, dilation=dilation, norm=norm[idx], nonlinear=nonlinear[idx], eps=eps)
            net.append(conv_block)

        self.net = nn.Sequential(*net)
        self.out_channels = _out_channels

    def forward(self, input):
        """
        Args:
            input: (batch_size, in_channels, H, W)
        Returns:
            output: (batch_size, out_channels, H, W), where `out_channels` is determined by growth_rate.
        """
        growth_rate, depth = self.growth_rate, self.depth

        for idx in range(depth):
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

class ConvBlock2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, norm=True, nonlinear='relu', eps=EPS):
        super().__init__()

        assert stride == 1, "`stride` is expected 1"

        self.kernel_size = _pair(kernel_size)
        self.dilation = _pair(dilation)

        self.norm = norm
        self.nonlinear = nonlinear

        if self.norm:
            if type(self.norm) is bool:
                name = 'BN'
            else:
                name = self.norm

            self.norm2d = choose_layer_norm(name, in_channels, n_dims=2, eps=eps)

        if self.nonlinear is not None:
            self.nonlinear2d = choose_nonlinear(self.nonlinear)

        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation)

    def forward(self, input):
        """
        Args:
            input (batch_size, in_channels, H, W)
        Returns:
            output (batch_size, out_channels, H, W)
        """
        Kh, Kw = self.kernel_size
        Dh, Dw = self.dilation

        padding_height = (Kh - 1) * Dh
        padding_width = (Kw - 1) * Dw
        padding_up = padding_height // 2
        padding_bottom = padding_height - padding_up
        padding_left = padding_width // 2
        padding_right = padding_width - padding_left

        x = input

        if self.norm:
            x = self.norm2d(x)

        if self.nonlinear:
            x = self.nonlinear2d(x)

        x = F.pad(x, (padding_left, padding_right, padding_up, padding_bottom))
        output = self.conv2d(x)

        return output

"""
    Quantization
"""
class QuantizableDenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, kernel_size, depth=None, dilated=False, norm=True, nonlinear='relu', eps=EPS):
        """
        Args:
            in_channels <int>: # of input channels
            growth_rate <int> or <list<int>>: # of output channels
            kernel_size <int> or <tuple<int>>: Kernel size
            dilated <bool> or <list<bool>>: Applies dilated convolution.
            norm <bool> or <list<bool>>: Applies batch normalization.
            nonlinear <str> or <list<str>>: Applies nonlinear function.
            depth <int>: If `growth_rate` is given by list, len(growth_rate) must be equal to `depth`.
        """
        super().__init__()

        if type(growth_rate) is int:
            assert depth is not None, "Specify `depth`"
            growth_rate = [growth_rate] * depth
        elif type(growth_rate) is list:
            if depth is not None:
                assert depth == len(growth_rate), "`depth` is different from `len(growth_rate)`"
            depth = len(growth_rate)
        else:
            raise ValueError("Not support growth_rate={}".format(growth_rate))

        if type(dilated) is bool:
            assert depth is not None, "Specify `depth`"
            dilated = [dilated] * depth
        elif type(dilated) is list:
            if depth is not None:
                assert depth == len(dilated), "`depth` is different from `len(dilated)`"
            depth = len(dilated)
        else:
            raise ValueError("Not support dilated={}".format(dilated))

        if type(norm) is bool:
            assert depth is not None, "Specify `depth`"
            norm = [norm] * depth
        elif type(norm) is list:
            if depth is not None:
                assert depth == len(norm), "`depth` is different from `len(norm)`"
            depth = len(norm)
        else:
            raise ValueError("Not support norm={}".format(norm))

        if type(nonlinear) is bool or type(nonlinear) is str:
            assert depth is not None, "Specify `depth`"
            nonlinear = [nonlinear] * depth
        elif type(nonlinear) is list:
            if depth is not None:
                assert depth == len(nonlinear), "`depth` is different from `len(nonlinear)`"
            depth = len(nonlinear)
        else:
            raise ValueError("Not support nonlinear={}".format(nonlinear))

        self.growth_rate = growth_rate
        self.depth = depth
        self.float_ops = nnq.FloatFunctional()

        net = []

        for idx in range(depth):
            if idx == 0:
                _in_channels = in_channels
            else:
                _in_channels = growth_rate[idx - 1]

            _out_channels = sum(growth_rate[idx:])

            if dilated[idx]:
                dilation = 2**idx
            else:
                dilation = 1

            conv_block = QuantizableConvBlock2d(_in_channels, _out_channels, kernel_size=kernel_size, stride=1, dilation=dilation, norm=norm[idx], nonlinear=nonlinear[idx], eps=eps)
            net.append(conv_block)

        self.net = nn.Sequential(*net)
        self.out_channels = _out_channels

    def forward(self, input):
        """
        Args:
            input: (batch_size, in_channels, H, W)
        Returns:
            output: (batch_size, out_channels, H, W), where `out_channels` is determined by growth_rate.
        """
        growth_rate, depth = self.growth_rate, self.depth

        for idx in range(depth):
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
                x_residual = self.float_ops.f_add(x_residual, x)

        output = x_residual

        return output

class QuantizableConvBlock2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, norm=True, nonlinear='relu', eps=EPS):
        super().__init__()

        assert stride == 1, "`stride` is expected 1"

        self.kernel_size = _pair(kernel_size)
        self.dilation = _pair(dilation)

        self.norm = norm
        self.nonlinear = nonlinear

        if self.norm:
            if type(self.norm) is bool:
                name = 'BN'
            else:
                name = self.norm

            self.norm2d = choose_layer_norm(name, in_channels, n_dims=2, eps=eps)

        if self.nonlinear is not None:
            self.nonlinear2d = choose_nonlinear(self.nonlinear)

        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation)

    def forward(self, input):
        """
        Args:
            input (batch_size, in_channels, H, W)
        Returns:
            output (batch_size, out_channels, H, W)
        """
        Kh, Kw = self.kernel_size
        Dh, Dw = self.dilation

        padding_height = (Kh - 1) * Dh
        padding_width = (Kw - 1) * Dw
        padding_up = padding_height // 2
        padding_bottom = padding_height - padding_up
        padding_left = padding_width // 2
        padding_right = padding_width - padding_left

        x = input

        if self.norm:
            x = self.norm2d(x)

        if self.nonlinear:
            x = self.nonlinear2d(x)

        x = F.pad(x, (padding_left, padding_right, padding_up, padding_bottom))
        output = self.conv2d(x)

        return output

def _test_dense_block():
    batch_size = 4
    n_bins, n_frames = 16, 64
    in_channels = 3
    growth_rate = 2
    kernel_size = (3, 3)
    depth = 4

    input = torch.randn(batch_size, in_channels, n_bins, n_frames)
    model = DenseBlock(in_channels, growth_rate, kernel_size=kernel_size, depth=depth)

    print(model)
    output = model(input)
    print(input.size(), output.size())
    print()

    growth_rate = 3
    model = DenseBlock(in_channels, growth_rate, kernel_size=kernel_size, depth=depth)

    print(model)
    output = model(input)
    print(input.size(), output.size())

def _test_down_dense_block():
    batch_size = 4
    n_bins, n_frames = 16, 64
    in_channels = 3
    growth_rate = 2
    kernel_size = (3, 3)
    down_scale = (2, 2)
    depth = 4

    input = torch.randn(batch_size, in_channels, n_bins, n_frames)
    model = DownSampleDenseBlock(in_channels, growth_rate, kernel_size=kernel_size, down_scale=down_scale, depth=depth)

    print(model)
    output, skip = model(input)
    print(input.size(), output.size(), skip.size())
    print()

    growth_rate = [3, 4, 5, 6]
    model = DownSampleDenseBlock(in_channels, growth_rate, kernel_size=kernel_size, down_scale=down_scale, depth=depth)

    print(model)
    output, skip = model(input)
    print(input.size(), output.size(), skip.size())

def _test_encoder():
    batch_size = 4
    n_bins, n_frames = 16, 64
    in_channels = 32

    growth_rate = [2, 3, 4]
    kernel_size = 3

    depth = [2, 2, 3]
    input = torch.randn(batch_size, in_channels, n_bins, n_frames)
    encoder = Encoder(in_channels, growth_rate, kernel_size, depth=depth)
    output, skip = encoder(input)

    print(encoder)
    print(input.size(), output.size())
    for _skip in skip:
        print(_skip.size())
    print()

    depth = 2
    encoder = Encoder(in_channels, growth_rate, kernel_size, depth=depth)
    output, skip = encoder(input)

    print(encoder)
    print(input.size(), output.size())

    for _skip in skip:
        print(_skip.size())

def _test_m_densenet_backbone():
    batch_size = 4
    n_bins, n_frames = 16, 64
    in_channels, num_features = 2, 32

    growth_rate = [2, 3, 4, 4, 2]
    kernel_size = 3

    dilated = [True, True, True, True, True]
    norm = [True, True, True, True, True]
    nonlinear = ['relu', 'relu', 'relu', 'relu', 'relu']
    depth = [3, 3, 4, 2, 2]
    input = torch.randn(batch_size, in_channels, n_bins, n_frames)

    model = MDenseNetBackbone(in_channels, num_features, growth_rate, kernel_size, dilated=dilated, norm=norm, nonlinear=nonlinear, depth=depth)

    print(model)

    output = model(input)

    print(input.size(), output.size())

def _test_m_densenet():
    config_path = "./data/m_densenet/paper.yaml"
    batch_size, in_channels, n_bins, n_frames = 4, 2, 1025, 256

    input = torch.randn(batch_size, in_channels, n_bins, n_frames)
    model = MDenseNet.build_from_config(config_path)

    output = model(input)

    print(model)
    print(input.size(), output.size())

if __name__ == '__main__':
    torch.manual_seed(111)

    print('='*10, "DenseBlock", '='*10)
    _test_dense_block()
    print()

    print('='*10, "DownSampleDenseBlock", '='*10)
    _test_down_dense_block()
    print()

    print('='*10, "Encoder", '='*10)
    _test_encoder()
    print()

    print('='*10, "MDenseNet backbone", '='*10)
    _test_m_densenet_backbone()
    print()

    print('='*10, "MDenseNet", '='*10)
    _test_m_densenet()
