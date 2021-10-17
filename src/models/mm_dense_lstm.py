import yaml

from models.mm_dense_rnn import MMDenseRNN

FULL = 'full'
EPS = 1e-12

class MMDenseLSTM(MMDenseRNN):
    def __init__(
        self,
        in_channels, num_features,
        growth_rate, hidden_channels, bottleneck_hidden_channels,
        kernel_size,
        bands=['low','middle'], sections=[512,513],
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
            in_channels, num_features, growth_rate, hidden_channels, bottleneck_hidden_channels,
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
            'hidden_channels': self.hidden_channels, 'bottleneck_hidden_channels': self.bottleneck_hidden_channels,
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
        bottleneck_hidden_channels = {
            band: config[band]['bottleneck_hidden_channels'] for band in bands + [FULL]
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
            growth_rate, hidden_channels, bottleneck_hidden_channels,
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
        hidden_channels, bottleneck_hidden_channels = config['hidden_channels'], config['bottleneck_hidden_channels']
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

        causal = config['causal']
        rnn_position = config['rnn_position']

        eps = config.get('eps') or EPS
        
        model = cls(
            in_channels, num_features,
            growth_rate, hidden_channels, bottleneck_hidden_channels,
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
            causal=causal,
            rnn_position=rnn_position,
            eps=eps
        )

        if load_state_dict:
            model.load_state_dict(config['state_dict'])
        
        return model

def _test_mm_dense_lstm():
    config_path = "./data/mm_dense_lstm/parallel.yaml"
    batch_size, in_channels, n_bins, n_frames = 4, 2, 1025, 256

    input = torch.randn(batch_size, in_channels, n_bins, n_frames)
    model = MMDenseLSTM.build_from_config(config_path)
    
    output = model(input)

    print(model)
    print(input.size(), output.size())

if __name__ == '__main__':
    import torch

    torch.manual_seed(111)

    print('='*10, "MMDenseLSTM", '='*10)
    _test_mm_dense_lstm()