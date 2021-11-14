import yaml
import torch
import torch.nn as nn

from models.mm_dense_rnn import MMDenseRNN

FULL = 'full'
SAMPLE_RATE_MUSDB18 = 44100
EPS = 1e-12
__pretrained_model_ids__ = {
    "musdb18": {
        SAMPLE_RATE_MUSDB18: {
            "paper": "1-2JGWMgVBdSj5zF9hl27jKhyX7GN-cOV"
        }
    },
}

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

        for key in modules.keys():
            module = modules[key]
            if not isinstance(module, MMDenseLSTM):
                raise ValueError("All modules must be MMDenseLSTM.")
            
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

class MMDenseLSTM(MMDenseRNN):
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

        if not task in __pretrained_model_ids__:
            raise KeyError("Invalid task ({}) is specified.".format(task))
            
        pretrained_model_ids_task = __pretrained_model_ids__[task]
        
        if task in ['musdb18']:
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