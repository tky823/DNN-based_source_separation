import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils_filterbank import choose_filterbank

EPS = 1e-12

class LSTMTasNet(nn.Module):
    def __init__(
        self,
        n_basis, kernel_size=80, stride=None, enc_basis=None, dec_basis=None,
        sep_num_layers=2, sep_hidden_channels=500,
        sep_dropout=0.3,
        mask_nonlinear='softmax',
        causal=False,
        n_sources=2,
        eps=EPS,
        **kwargs
    ):
        super().__init__()
        
        if stride is None:
            stride = kernel_size // 2
        
        assert kernel_size % stride == 0, "kernel_size is expected divisible by stride"
        assert enc_basis in ['trainable', 'trainableGated']  and dec_basis == 'trainable', "enc_basis is expected 'trainable' or 'trainableGated'. dec_basis is expected 'trainable'."
        
        if 'in_channels' in kwargs:
            self.in_channels = kwargs['in_channels']
        else:
            self.in_channels = 1
        self.n_basis = n_basis
        self.kernel_size, self.stride = kernel_size, stride
        self.enc_basis, self.dec_basis = enc_basis, dec_basis

        # Separator configuration
        self.sep_num_layers = sep_num_layers
        self.sep_hidden_channels = sep_hidden_channels
        self.sep_dropout = sep_dropout
        self.causal = causal
        self.mask_nonlinear = mask_nonlinear
        self.n_sources = n_sources
        self.eps = eps
        
        encoder, decoder = choose_filterbank(n_basis, kernel_size=kernel_size, stride=stride, enc_basis=enc_basis, dec_basis=dec_basis, **kwargs)
        
        self.encoder = encoder
        self.separator = Separator(
            n_basis, num_layers=sep_num_layers,
            hidden_channels=sep_hidden_channels,
            dropout=sep_dropout,
            causal=causal,
            mask_nonilnear=mask_nonlinear,
            n_sources=n_sources
        )
        self.decoder = decoder
    
    def forward(self, input):
        """
        Args:
            input (batch_size, 1, T)
        Returns:
            output (batch_size, n_sources, T)
        """
        output, _ = self.extract_latent(input)
        
        return output
    
    def extract_latent(self, input):
        """
        Args:
            input (batch_size, 1, T)
        Returns:
            output (batch_size, n_sources, T)
            latent (batch_size, n_sources, n_basis, T'), where T' = (T-K)//S+1
        """
        n_sources = self.n_sources
        n_basis = self.n_basis
        kernel_size, stride = self.kernel_size, self.stride
        
        n_dims = input.dim()

        if n_dims == 3:
            batch_size, C_in, T = input.size()
            assert C_in == 1, "input.size() is expected (?, 1, ?), but given {}".format(input.size())
        elif n_dims == 4:
            batch_size, C_in, n_mics, T = input.size()
            assert C_in == 1, "input.size() is expected (?, 1, ?, ?), but given {}".format(input.size())
            input = input.view(batch_size, n_mics, T)
        else:
            raise ValueError("Not support {} dimension input".format(n_dims))
        
        padding = (stride - (T - kernel_size) % stride) % stride
        padding_left = padding // 2
        padding_right = padding - padding_left

        input = F.pad(input, (padding_left, padding_right))
        w = self.encoder(input)

        if torch.is_complex(w):
            amplitude, phase = torch.abs(w), torch.angle(w)
            mask = self.separator(amplitude)
            amplitude, phase = amplitude.unsqueeze(dim=1), phase.unsqueeze(dim=1)
            w_hat = amplitude * mask * torch.exp(1j * phase)
        else:
            mask = self.separator(w)
            w = w.unsqueeze(dim=1)
            w_hat = w * mask
        
        latent = w_hat
        w_hat = w_hat.view(batch_size*n_sources, n_basis, -1)
        x_hat = self.decoder(w_hat)
        if n_dims == 3:
            x_hat = x_hat.view(batch_size, n_sources, -1)
        else: # n_dims == 4
            x_hat = x_hat.view(batch_size, n_sources, n_mics, -1)
        
        output = F.pad(x_hat, (-padding_left, -padding_right))
        
        return output, latent
    
    @classmethod
    def build_model(cls, model_path, load_state_dict=False):
        config = torch.load(model_path, map_location=lambda storage, loc: storage)
        
        in_channels = config.get('in_channels') or 1
        n_basis = config.get('n_bases') or config['n_basis']
        kernel_size, stride = config['kernel_size'], config['stride']
        enc_basis, dec_basis = config.get('enc_bases') or config['enc_basis'], config.get('dec_bases') or config['dec_basis']
        enc_nonlinear = config.get('enc_nonlinear')
        
        sep_num_layers = config['sep_num_layers']
        sep_hidden_channels = config['sep_hidden_channels']
        sep_dropout = config['sep_dropout']
        
        causal = config['causal']
        mask_nonlinear = config['mask_nonlinear']
        
        n_sources = config['n_sources']
        
        eps = config.get('eps') or EPS
        
        model = cls(
            n_basis, in_channels=in_channels, kernel_size=kernel_size, stride=stride, enc_basis=enc_basis, dec_basis=dec_basis, enc_nonlinear=enc_nonlinear,
            sep_num_layers=sep_num_layers, sep_hidden_channels=sep_hidden_channels,
            sep_dropout=sep_dropout,
            mask_nonlinear=mask_nonlinear,
            causal=causal,
            n_sources=n_sources,
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
    
    def get_config(self):
        config = {
            'in_channels': self.in_channels,
            'n_basis': self.n_basis,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'enc_basis': self.enc_basis,
            'dec_basis': self.dec_basis,
            'sep_num_layers': self.sep_num_layers,
            'sep_hidden_channels': self.sep_hidden_channels,
            'sep_dropout': self.sep_dropout,
            'causal': self.causal,
            'mask_nonlinear': self.mask_nonlinear,
            'n_sources': self.n_sources,
            'eps': self.eps
        }
        
        return config

"""
    Modules for LSTM-TasNet
"""

class Separator(nn.Module):
    def __init__(
        self,
        n_basis, num_layers, hidden_channels,
        dropout=0.3,
        causal=False,
        mask_nonilnear='softmax',
        n_sources=2
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.n_basis, self.n_sources = n_basis, n_sources

        if causal:
            num_directions = 1
            bidirectional = False
        else:
            num_directions = 2
            bidirectional = True
        
        self.rnn = nn.LSTM(n_basis, hidden_channels, num_layers=num_layers, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        self.fc = nn.Linear(num_directions * hidden_channels, n_sources * n_basis)
        if mask_nonilnear == 'sigmoid':
            self.mask_nonlinear = nn.Sigmoid()
        elif mask_nonilnear == 'softmax':
            self.mask_nonlinear = nn.Softmax(dim=1)
        else:
            raise ValueError("Only supports sigmoid and softmax, but given {}.".format(mask_nonilnear))
    
    def forward(self, input):
        n_basis, n_sources = self.n_basis, self.n_sources
        
        batch_size, _, n_frames = input.size()
        
        self.rnn.flatten_parameters()
        
        x = input.permute(0, 2, 1).contiguous() # (batch_size, n_frames, n_basis)
        x, (_, _) = self.rnn(x)
        x = self.fc(x) # (batch_size, n_frames, n_sources * n_basis)
        x = x.view(batch_size, n_frames, n_sources, n_basis)
        x = x.permute(0, 2, 3, 1).contiguous() # (batch_size, n_sources, n_basis, n_frames)
        output = self.mask_nonlinear(x)

        return output

def _test_tasnet():
    batch_size = 2
    C = 1
    T = 64
    kernel_size, stride = 8, 2
    repeat = 2
    n_basis = kernel_size * repeat * 2
    
    input = torch.randn((batch_size, C, T), dtype=torch.float)

    # LSTM-TasNet configuration
    sep_num_blocks, sep_num_layers, sep_hidden_channels = 2, 2, 32
    n_sources = 3
    
    # Non causal
    print("-"*10, "Non causal", "-"*10)
    causal = False

    model = LSTMTasNet(
        n_basis, kernel_size=kernel_size, stride=stride,
        enc_basis='trainable', dec_basis='trainable', enc_nonlinear=None,
        sep_num_blocks=sep_num_blocks, sep_num_layers=sep_num_layers, sep_hidden_channels=sep_hidden_channels,
        causal=causal, n_sources=n_sources
    )
    print(model)
    print("# Parameters: {}".format(model.num_parameters))
    
    output = model(input)
    print(input.size(), output.size())
    print()
    
    # Causal
    print("-"*10, "Causal", "-"*10)
    causal = True
    
    model = LSTMTasNet(
        n_basis, kernel_size=kernel_size, stride=stride,
        enc_basis='trainable', dec_basis='trainable', enc_nonlinear=None,
        sep_num_blocks=sep_num_blocks, sep_num_layers=sep_num_layers, sep_hidden_channels=sep_hidden_channels,
        causal=causal, n_sources=n_sources
    )
    print(model)
    print("# Parameters: {}".format(model.num_parameters))

    output = model(input)
    print(input.size(), output.size())

if __name__ == '__main__':
    print("="*10, "LSTM-TasNet", "="*10)
    _test_tasnet()
