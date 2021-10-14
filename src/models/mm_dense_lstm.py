import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from utils.utils_m_densenet import choose_layer_norm
from utils.utils_dense_lstm import choose_dense_rnn_block
from models.transform import BandSplit
from models.glu import GLU2d
from models.m_densenet import DenseBlock

"""
Reference: MMDenseLSTM: An efficient combination of convolutional and recurrent neural networks for audio source separation
See https://ieeexplore.ieee.org/document/8521383
"""

FULL = 'full'
EPS = 1e-12

class MMDenseRNN(nn.Module):
    """
    Multi-scale Multi-band Dense RNN
    """
    def __init__(
        self,
        in_channels, num_features,
        growth_rate,
        kernel_size,
        bands=['low','middle'], sections=[512,513],
        scale=(2,2),
        dilated=False, norm=True, nonlinear='relu',
        depth=None,
        growth_rate_final=None,
        kernel_size_final=None,
        dilated_final=False,
        norm_final=True, nonlinear_final='relu',
        depth_final=None,
        rnn_position='parallel',
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
            
            net[band] = MMDenseRNNBackbone(
                in_channels, num_features[band], growth_rate[band],
                kernel_size[band], scale=scale[band],
                dilated=dilated[band], norm=norm[band], nonlinear=nonlinear[band],
                depth=depth[band],
                rnn_position=rnn_position,
                out_channels=_out_channels,
                eps=eps
            )
        
        net[FULL] = MMDenseRNNBackbone(
            in_channels, num_features[FULL], growth_rate[FULL],
            kernel_size[FULL], scale=scale[FULL],
            dilated=dilated[FULL], norm=norm[FULL], nonlinear=nonlinear[FULL],
            depth=depth[FULL],
            rnn_position=rnn_position,
            eps=eps
        )

        self.net = nn.ModuleDict(net)

        _in_channels = out_channels + growth_rate[FULL][-1] # channels for 'low' & 'middle' + channels for 'full'
        
        if kernel_size_final is None:
            kernel_size_final = kernel_size

        self.dense_block = choose_dense_rnn_block(rnn_position, _in_channels, growth_rate_final, kernel_size_final, dilated=dilated_final, depth=depth_final, norm=norm_final, nonlinear=nonlinear_final, eps=eps)
        self.norm2d = choose_layer_norm('BN', growth_rate_final, n_dims=2, eps=eps) # nn.BatchNorm2d
        self.glu2d = GLU2d(growth_rate_final, in_channels, kernel_size=(1,1), stride=(1,1))
        self.relu2d = nn.ReLU()

        self.scale_in, self.bias_in = nn.Parameter(torch.Tensor(sum(sections),), requires_grad=True), nn.Parameter(torch.Tensor(sum(sections),), requires_grad=True)
        self.scale_out, self.bias_out = nn.Parameter(torch.Tensor(sum(sections),), requires_grad=True), nn.Parameter(torch.Tensor(sum(sections),), requires_grad=True)

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

        self.rnn_position = rnn_position

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
        eps = self.eps

        if sum(sections) == n_bins:
            x_valid, x_invalid = input, None
        else:
            sections = [sum(sections), n_bins - sum(sections)]
            x_valid, x_invalid = torch.split(input, sections, dim=2)

        x_valid = (x_valid - self.bias_in.unsqueeze(dim=1)) / (torch.abs(self.scale_in.unsqueeze(dim=1)) + eps)

        x = self.band_split(x_valid)

        x_bands = []
        for band, x_band in zip(bands, x):
            x_band = self.net[band](x_band)
            x_bands.append(x_band)
        x_bands = torch.cat(x_bands, dim=2)
    
        x_full = self.net[FULL](x_valid)

        x = torch.cat([x_bands, x_full], dim=1)

        x = self.dense_block(x)
        x = self.norm2d(x)
        x = self.glu2d(x)
        x = self.scale_out.unsqueeze(dim=1) * x + self.bias_out.unsqueeze(dim=1)
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
            'dilated': self.dilated, 'norm': self.norm, 'nonlinear': self.nonlinear,
            'depth': self.depth,
            'growth_rate_final': self.growth_rate_final,
            'kernel_size_final': self.kernel_size_final,
            'dilated_final': self.dilated_final,
            'depth_final': self.depth_final,
            'norm_final': self.norm_final, 'nonlinear_final': self.nonlinear_final,
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

        growth_rate_final = config['final']['growth_rate']
        kernel_size_final = config['final']['kernel_size']
        dilated_final = config['final']['dilated']
        depth_final = config['final']['depth']
        norm_final, nonlinear_final = config['final']['norm'], config['final']['nonlinear']

        rnn_position = config['rnn_position']

        eps = config.get('eps') or EPS

        model = cls(
            in_channels, num_features,
            growth_rate,
            kernel_size,
            bands=bands, sections=sections,
            scale=scale,
            dilated=dilated, norm=norm, nonlinear=nonlinear,
            depth=depth,
            growth_rate_final=growth_rate_final,
            kernel_size_final=kernel_size_final,
            dilated_final=dilated_final,
            depth_final=depth_final,
            norm_final=norm_final, nonlinear_final=nonlinear_final,
            rnn_position=rnn_position,
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

        dilated, norm, nonlinear = config['dilated'], config['norm'], config['nonlinear']
        depth = config['depth']

        growth_rate_final = config['growth_rate_final']
        kernel_size_final = config['kernel_size_final']
        dilated_final = config['dilated_final']
        depth_final = config['depth_final']
        norm_final, nonlinear_final = config['norm_final'] or True, config['nonlinear_final']

        rnn_position = config['rnn_position']

        eps = config.get('eps') or EPS
        
        model = cls(
            in_channels, num_features,
            growth_rate,
            kernel_size,
            bands=bands, sections=sections,
            scale=scale,
            dilated=dilated, norm=norm, nonlinear=nonlinear,
            depth=depth,
            growth_rate_final=growth_rate_final,
            kernel_size_final=kernel_size_final,
            dilated_final=dilated_final,
            depth_final=depth_final,
            norm_final=norm_final, nonlinear_final=nonlinear_final,
            rnn_position=rnn_position,
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

class MMDenseRNNBackbone(nn.Module):
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
        self.conv2d = nn.Conv2d(in_channels, num_features, kernel_size, stride=(1,1))

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
            downsample_block = DownSampleDenseRNNBlock(_in_channels, growth_rate[idx], kernel_size=kernel_size, down_scale=down_scale, dilated=dilated[idx], norm=norm[idx], nonlinear=nonlinear[idx], depth=depth[idx], eps=eps)
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
            upsample_block = UpSampleDenseRNNBlock(_in_channels, skip_channels[idx], growth_rate[idx], kernel_size=kernel_size, up_scale=up_scale, dilated=dilated[idx], norm=norm[idx], nonlinear=nonlinear[idx], depth=depth[idx], eps=eps)
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

class DownSampleDenseRNNBlock(nn.Module):
    """
    DenseRNNBlock + down sample
    """
    def __init__(self, in_channels, growth_rate, kernel_size=(3,3), down_scale=(2,2), dilated=False, norm=True, nonlinear='relu', depth=None, eps=EPS):
        super().__init__()

        self.down_scale = _pair(down_scale)

        self.dense_rnn_block = choose_dense_rnn_block('parallel', in_channels, growth_rate, kernel_size, dilated=dilated, norm=norm, nonlinear=nonlinear, depth=depth, eps=eps)
        self.downsample2d = nn.AvgPool2d(kernel_size=self.down_scale, stride=self.down_scale)
    
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
        
        x = self.dense_rnn_block(input)
        skip = x
        skip = F.pad(skip, (-padding_left, -padding_right, -padding_top, -padding_bottom))

        output = self.downsample2d(x)

        return output, skip

class UpSampleDenseRNNBlock(nn.Module):
    """
    DenseRNNBlock + up sample
    """
    def __init__(self, in_channels, skip_channels, growth_rate, kernel_size=(2,2), up_scale=(2,2), dilated=False, norm=True, nonlinear='relu', depth=None, eps=EPS):
        super().__init__()

        self.norm2d = choose_layer_norm('BN', in_channels, n_dims=2, eps=eps) # nn.BatchNorm2d
        self.upsample2d = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=up_scale, stride=up_scale)
        self.dense_rnn_block = choose_dense_rnn_block('parallel', in_channels + skip_channels, growth_rate, kernel_size, dilated=dilated, norm=norm, nonlinear=nonlinear, depth=depth, eps=eps)
    
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

        output = self.dense_rnn_block(x)

        return output

if __name__ == '__main__':
    torch.manual_seed(111)