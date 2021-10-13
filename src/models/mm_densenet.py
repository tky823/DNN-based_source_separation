import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils_m_densenet import choose_layer_norm
from models.transform import BandSplit
from models.glu import GLU2d
from models.m_densenet import MDenseNetBackbone, DenseBlock

FULL = 'full'
EPS = 1e-12

class ParallelMMDenseNet(nn.Module):
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
            if not isinstance(module, MMDenseNet):
                raise ValueError("All modules must be D3Net.")
            
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

class MMDenseNet(nn.Module):
    """
    Multi-scale Multi-band DenseNet
    """
    def __init__(
        self,
        in_channels, num_features,
        growth_rate,
        kernel_size,
        bands=['low','middle'], sections=[256,1344],
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
            
            net[band] = MDenseNetBackbone(
                in_channels, num_features[band], growth_rate[band],
                kernel_size[band], scale=scale[band],
                dilated=dilated[band], norm=norm[band], nonlinear=nonlinear[band],
                depth=depth[band],
                out_channels=_out_channels,
                eps=eps
            )
        net[FULL] = MDenseNetBackbone(
            in_channels, num_features[FULL], growth_rate[FULL],
            kernel_size[FULL], scale=scale[FULL],
            dilated=dilated[FULL], norm=norm[FULL], nonlinear=nonlinear[FULL],
            depth=depth[FULL],
            eps=eps
        )

        self.net = nn.ModuleDict(net)

        _in_channels = out_channels + growth_rate[FULL][-1] # channels for 'low' & 'middle' + channels for 'full'
        
        if kernel_size_final is None:
            kernel_size_final = kernel_size

        self.d2block = DenseBlock(_in_channels, growth_rate_final, kernel_size_final, dilated=dilated_final, depth=depth_final, norm=norm_final, nonlinear=nonlinear_final, eps=eps)
        self.norm2d = choose_layer_norm('BN', growth_rate_final, n_dims=2, eps=eps) # nn.BatchNorm2d
        self.glu2d = GLU2d(growth_rate_final, in_channels, kernel_size=(1,1), stride=(1,1))
        self.relu2d = nn.ReLU()

        self.scale_in, self.bias_in = nn.Parameter(torch.Tensor(sum(sections),)), nn.Parameter(torch.Tensor(sum(sections),))
        self.scale_out, self.bias_out = nn.Parameter(torch.Tensor(sum(sections),)), nn.Parameter(torch.Tensor(sum(sections),))

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
        sections = self.sections
        n_bins = input.size(2)
        eps = self.eps

        if sum(sections) == n_bins:
            x_valid, x_invalid = input, None
        else:
            sections = [sum(sections), n_bins - sum(sections)]
            x_valid, x_invalid = torch.split(input, sections, dim=2)

        x = (x_valid - self.bias_in.unsqueeze(dim=1)) / (torch.abs(self.scale_in.unsqueeze(dim=1)) + eps)

        x = self.net(x)
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

        eps = config.get('eps') or EPS

        model = cls(
            in_channels, num_features,
            growth_rate,
            kernel_size,
            bands=bands, sections=sections,
            scale=scale,
            depth=depth, dilated=dilated, norm=norm, nonlinear=nonlinear,
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

        eps = config.get('eps') or EPS
        
        model = cls(
            in_channels, num_features,
            growth_rate,
            kernel_size,
            max_bin=max_bin,
            scale=scale,
            dilated=dilated, norm=norm, nonlinear=nonlinear,
            depth=depth,
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
