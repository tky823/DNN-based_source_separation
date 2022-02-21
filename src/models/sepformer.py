import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.filterbank import choose_filterbank
from utils.model import choose_nonlinear
from utils.tasnet import choose_layer_norm
from models.transform import Segment1d, OverlapAdd1d
from models.transformer import PositionalEncoding
from models.gtu import GTU1d

EPS = 1e-12

class SepFormer(nn.Module):
    pretrained_model_ids = {
        "wsj0-mix": {
            8000: {
                2: "1-9pOv2B612IykvpA6kaGZSg4AUQPnoCg",
                3: "1-Rz31CGWVVzYVHXgIdp7Tuc0__K2SCPs"
            }
        }
    }
    def __init__(
        self,
        n_basis, kernel_size, stride=None, enc_basis=None, dec_basis=None,
        sep_bottleneck_channels=None,
        sep_chunk_size=250, sep_hop_size=125,
        sep_num_blocks=2,
        sep_num_layers_intra=8, sep_num_layers_inter=8, sep_num_heads_intra=8, sep_num_heads_inter=8,
        sep_d_ff_intra=1024, sep_d_ff_inter=1024,
        sep_norm=True, sep_nonlinear='relu', sep_dropout=1e-1, mask_nonlinear='relu',
        causal=True,
        n_sources=2,
        eps=EPS,
        **kwargs
    ):
        super().__init__()

        if stride is None:
            stride = kernel_size // 2

        assert kernel_size % stride == 0, "kernel_size is expected divisible by stride"

        # Encoder-decoder
        self.in_channels = kwargs.get('in_channels', 1)
        self.n_basis = n_basis
        self.kernel_size, self.stride = kernel_size, stride
        self.enc_basis, self.dec_basis = enc_basis, dec_basis

        if enc_basis == 'trainable' and not dec_basis == 'pinv':    
            self.enc_nonlinear = kwargs['enc_nonlinear']
        else:
            self.enc_nonlinear = None

        if enc_basis in ['Fourier', 'trainableFourier', 'trainableFourierTrainablePhase'] or dec_basis in ['Fourier', 'trainableFourier', 'trainableFourierTrainablePhase']:
            self.window_fn = kwargs['window_fn']
            self.enc_onesided, self.enc_return_complex = kwargs['enc_onesided'], kwargs['enc_return_complex']
        else:
            self.window_fn = None
            self.enc_onesided, self.enc_return_complex = None, None

        # Separator configuration
        if sep_bottleneck_channels is None:
            sep_bottleneck_channels = n_basis

        self.sep_bottleneck_channels = sep_bottleneck_channels
        self.sep_chunk_size, self.sep_hop_size = sep_chunk_size, sep_hop_size
        self.sep_num_blocks = sep_num_blocks

        self.sep_num_layers_intra, self.sep_num_layers_inter = sep_num_layers_intra, sep_num_layers_inter
        self.sep_num_heads_intra, self.sep_num_heads_inter = sep_num_heads_intra, sep_num_heads_inter
        self.sep_d_ff_intra, self.sep_d_ff_inter = sep_d_ff_intra, sep_d_ff_inter,

        self.causal = causal
        self.sep_norm, self.sep_dropout = sep_norm, sep_dropout
        self.sep_nonlinear, self.mask_nonlinear = sep_nonlinear, mask_nonlinear

        self.n_sources = n_sources
        self.eps = eps

        # Network configuration
        encoder, decoder = choose_filterbank(n_basis, kernel_size=kernel_size, stride=stride, enc_basis=enc_basis, dec_basis=dec_basis, **kwargs)

        self.encoder = encoder
        self.separator = Separator(
            n_basis, sep_bottleneck_channels,
            chunk_size=sep_chunk_size, hop_size=sep_hop_size,
            num_blocks=sep_num_blocks,
            num_layers_intra=sep_num_layers_intra, num_layers_inter=sep_num_layers_inter, num_heads_intra=sep_num_heads_intra, num_heads_inter=sep_num_heads_inter,
            d_ff_intra=sep_d_ff_intra, d_ff_inter=sep_d_ff_inter,
            norm=sep_norm, nonlinear=sep_nonlinear, dropout=sep_dropout, mask_nonlinear=mask_nonlinear,
            causal=causal,
            n_sources=n_sources,
            eps=eps
        )
        self.decoder = decoder

    def forward(self, input):
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

    def get_config(self):
        config = {
            'in_channels': self.in_channels,
            'n_basis': self.n_basis,
            'kernel_size': self.kernel_size, 'stride': self.stride,
            'enc_basis': self.enc_basis, 'dec_basis': self.dec_basis, 'enc_nonlinear': self.enc_nonlinear,
            'enc_onesided': self.enc_onesided, 'enc_return_complex': self.enc_return_complex, 'window_fn': self.window_fn,
            'sep_bottleneck_channels': self.sep_bottleneck_channels,
            'sep_chunk_size': self.sep_chunk_size, 'sep_hop_size': self.sep_hop_size,
            'sep_num_blocks': self.sep_num_blocks,
            'sep_num_layers_intra': self.sep_num_layers_intra, 'sep_num_layers_inter': self.sep_num_layers_inter,
            'sep_num_heads_intra': self.sep_num_heads_intra, 'sep_num_heads_inter': self.sep_num_heads_inter,
            'sep_d_ff_intra': self.sep_d_ff_intra, 'sep_d_ff_inter': self.sep_d_ff_inter,
            'sep_norm': self.sep_norm, 'sep_nonlinear': self.sep_nonlinear, 'sep_dropout': self.sep_dropout, 'mask_nonlinear': self.mask_nonlinear,
            'causal': self.causal,
            'n_sources': self.n_sources,
            'eps': self.eps
        }

        return config

    @classmethod
    def build_model(cls, model_path, load_state_dict=False):
        config = torch.load(model_path, map_location=lambda storage, loc: storage)

        in_channels = config['in_channels']
        n_basis = config['n_basis']
        kernel_size, stride = config['kernel_size'], config['stride']
        enc_basis, dec_basis = config['enc_basis'], config['dec_basis']
        enc_nonlinear = config['enc_nonlinear']
        enc_onesided, enc_return_complex = config['enc_onesided'], config['enc_return_complex']
        window_fn = config['window_fn']

        sep_bottleneck_channels = config['sep_bottleneck_channels']
        sep_chunk_size, sep_hop_size = config['sep_chunk_size'], config['sep_hop_size']
        sep_num_blocks = config['sep_num_blocks']
        sep_num_layers_intra, sep_num_layers_inter = config['sep_num_layers_intra'], config['sep_num_layers_inter']
        sep_num_heads_intra, sep_num_heads_inter = config['sep_num_heads_intra'], config['sep_num_heads_inter']
        sep_d_ff_intra, sep_d_ff_inter = config['sep_d_ff_intra'], config['sep_d_ff_inter']

        sep_norm = config['sep_norm']
        sep_dropout = config['sep_dropout']
        sep_nonlinear, mask_nonlinear = config['sep_nonlinear'], config['mask_nonlinear']

        causal = config['causal']
        n_sources = config['n_sources']

        eps = config['eps']

        model = cls(
            n_basis, in_channels=in_channels,
            kernel_size=kernel_size, stride=stride,
            enc_basis=enc_basis, dec_basis=dec_basis, enc_nonlinear=enc_nonlinear,
            window_fn=window_fn, enc_onesided=enc_onesided, enc_return_complex=enc_return_complex,
            sep_bottleneck_channels=sep_bottleneck_channels,
            sep_chunk_size=sep_chunk_size, sep_hop_size=sep_hop_size,
            sep_num_blocks=sep_num_blocks, sep_num_layers_intra=sep_num_layers_intra, sep_num_layers_inter=sep_num_layers_inter,
            sep_num_heads_intra=sep_num_heads_intra, sep_num_heads_inter=sep_num_heads_inter,
            sep_d_ff_intra=sep_d_ff_intra, sep_d_ff_inter=sep_d_ff_inter,
            sep_norm=sep_norm, sep_nonlinear=sep_nonlinear, sep_dropout=sep_dropout, mask_nonlinear=mask_nonlinear,
            causal=causal,
            n_sources=n_sources,
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
            model_choice = kwargs.get('model_choice') or 'best'

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

        model = cls.build_model(model_path, load_state_dict=load_state_dict)

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

class Separator(nn.Module):
    def __init__(
        self,
        num_features, bottleneck_channels,
        chunk_size=250, hop_size=125,
        num_blocks=2, num_layers_intra=8, num_layers_inter=8,
        num_heads_intra=8, num_heads_inter=8,
        d_ff_intra=1024, d_ff_inter=1024,
        norm=True, nonlinear='relu', dropout=1e-1, mask_nonlinear='relu',
        causal=False,
        n_sources=2,
        eps=EPS
    ):
        super().__init__()

        self.num_features, self.n_sources = num_features, n_sources
        self.chunk_size, self.hop_size = chunk_size, hop_size
        self.norm = norm

        norm_name = 'cLN' if causal else 'gLN'
        self.norm1d = choose_layer_norm(norm_name, num_features, causal=causal, eps=eps)
        self.bottleneck_conv1d_in = nn.Conv1d(num_features, bottleneck_channels, kernel_size=1, stride=1)

        self.segment1d = Segment1d(chunk_size, hop_size)
        self.dptransformer = SepFormerBackbone(
            num_blocks=num_blocks, num_layers_intra=num_layers_intra, num_layers_inter=num_layers_inter,
            num_heads_intra=num_heads_intra, num_heads_inter=num_heads_inter,
            d_intra=bottleneck_channels, d_inter=bottleneck_channels, d_ff_intra=d_ff_intra, d_ff_inter=d_ff_inter,
            norm=norm, dropout=dropout, nonlinear=nonlinear,
            causal=causal,
            eps=eps
        )
        self.overlap_add1d = OverlapAdd1d(chunk_size, hop_size)

        self.prelu = nn.PReLU()
        self.map = nn.Conv1d(bottleneck_channels, n_sources * num_features, kernel_size=1, stride=1)
        self.gtu = GTU1d(num_features, num_features, kernel_size=1, stride=1)
        self.bottleneck_conv1d_out = nn.Conv1d(num_features, num_features, kernel_size=1, stride=1)

        if mask_nonlinear in ['relu', 'sigmoid']:
            kwargs = {}
        elif mask_nonlinear == 'softmax':
            kwargs = {
                'dim': 1
            }
        else:
            raise ValueError("Cannot support {}".format(mask_nonlinear))

        self.mask_nonlinear = choose_nonlinear(mask_nonlinear, **kwargs)

    def forward(self, input):
        """
        Args:
            input (batch_size, num_features, n_frames)
        Returns:
            output (batch_size, n_sources, num_features, n_frames)
        """
        num_features, n_sources = self.num_features, self.n_sources
        chunk_size, hop_size = self.chunk_size, self.hop_size
        batch_size, num_features, n_frames = input.size()

        padding = (hop_size - (n_frames - chunk_size) % hop_size) % hop_size
        padding_left = padding // 2
        padding_right = padding - padding_left

        x = self.norm1d(input)
        x = self.bottleneck_conv1d_in(x)
        x = F.pad(x, (padding_left, padding_right))
        x = self.segment1d(x)
        x = self.dptransformer(x)
        x = self.overlap_add1d(x)
        x = F.pad(x, (-padding_left, -padding_right))
        x = self.prelu(x)
        x = self.map(x) # (batch_size, n_sources * C, n_frames)
        x = x.view(batch_size * n_sources, num_features, n_frames) # (batch_size * n_sources, num_features, n_frames)
        x = self.gtu(x) # (batch_size * n_sources, num_features, n_frames)
        x = self.bottleneck_conv1d_out(x) # (batch_size * n_sources, num_features, n_frames)
        x = self.mask_nonlinear(x) # (batch_size * n_sources, num_features, n_frames)
        output = x.view(batch_size, n_sources, num_features, n_frames)

        return output

class SepFormerBackbone(nn.Module):
    def __init__(
        self,
        num_blocks=2, num_layers_intra=8, num_layers_inter=8,
        num_heads_intra=8, num_heads_inter=8,
        d_intra=256, d_inter=256, d_ff_intra=1024, d_ff_inter=1024,
        norm=True, dropout=1e-1, nonlinear='relu', causal=False,
        eps=EPS
    ):
        super().__init__()

        # Network confguration
        net = []

        for _ in range(num_blocks):
            module = SepFormerBlock(
                num_layers_intra=num_layers_intra, num_layers_inter=num_layers_inter,
                num_heads_intra=num_heads_intra, num_heads_inter=num_heads_inter,
                d_intra=d_intra, d_inter=d_inter, d_ff_intra=d_ff_intra, d_ff_inter=d_ff_inter,
                norm=norm, dropout=dropout, nonlinear=nonlinear,
                causal=causal,
                eps=eps
            )
            net.append(module)

        self.net = nn.Sequential(*net)

    def forward(self, input):
        """
        Args:
            input (batch_size, num_features, S, chunk_size)
        Returns:
            output (batch_size, num_features, S, chunk_size)
        """
        output = self.net(input)

        return output

class SepFormerBlock(nn.Module):
    def __init__(
        self,
        num_layers_intra=8, num_layers_inter=8,
        num_heads_intra=8, num_heads_inter=8,
        d_intra=256, d_inter=256, d_ff_intra=1024, d_ff_inter=1024,
        norm=True, dropout=1e-1, nonlinear='relu',
        causal=False,
        eps=EPS
    ):
        super().__init__()

        self.intra_transformer = IntraTransformer(
            d_intra,
            num_layers=num_layers_intra, num_heads=num_heads_intra, d_ff=d_ff_intra,
            norm=norm, dropout=dropout, nonlinear=nonlinear,
            eps=eps
        )
        self.inter_transformer = InterTransformer(
            d_inter,
            num_layers=num_layers_inter, num_heads=num_heads_inter, d_ff=d_ff_inter,
            norm=norm, dropout=dropout, nonlinear=nonlinear, causal=causal,
            eps=eps
        )

    def forward(self, input):
        """
        Args:
            input (batch_size, num_features, S, chunk_size)
        Returns:
            output (batch_size, num_features, S, chunk_size)
        """
        x = self.intra_transformer(input)
        output = self.inter_transformer(x)

        return output

class IntraTransformer(nn.Module):
    def __init__(self, num_features, num_layers=8, num_heads=8, d_ff=1024, norm=True, nonlinear='relu', dropout=1e-1, norm_first=False, eps=EPS):
        super().__init__()

        self.num_features = num_features

        if isinstance(norm, int):
            if norm:
                norm_name = 'gLN'
                layer_norm = LayerNormWrapper(norm_name, num_features, causal=False, batch_first=False, eps=eps)
            else:
                layer_norm = None
        else:
            norm_name = norm
            layer_norm = LayerNormWrapper(norm_name, num_features, causal=False, batch_first=False, eps=eps)

        self.positional_encoding = PositionalEncoding(num_features, batch_first=False)
        encoder_layer = nn.TransformerEncoderLayer(num_features, num_heads, d_ff, dropout=dropout, activation=nonlinear, layer_norm_eps=eps, batch_first=False, norm_first=norm_first)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, norm=layer_norm)

    def forward(self, input):
        """
        Args:
            input (batch_size, num_features, S, chunk_size)
        Returns:
            output (batch_size, num_features, S, chunk_size)
        """
        num_features = self.num_features
        batch_size, _, S, chunk_size = input.size()

        residual = input
        x = input.permute(3, 0, 2, 1).contiguous() # (batch_size, num_features, S, chunk_size) -> (chunk_size, batch_size, S, num_features)
        x = x.view(chunk_size, batch_size * S, num_features) # (chunk_size, batch_size, S, num_features) -> (chunk_size, batch_size * S, num_features)
        embedding = self.positional_encoding(x)
        x = x + embedding
        x = self.transformer(x) # (chunk_size, batch_size * S, num_features)
        x = x.view(chunk_size, batch_size, S, num_features) # (chunk_size, batch_size * S, num_features) -> (chunk_size, batch_size, S, num_features)
        x = x.permute(1, 3, 2, 0).contiguous() # (chunk_size, batch_size, S, num_features) -> (batch_size, num_features, S, chunk_size)
        output = x + residual

        return output

class InterTransformer(nn.Module):
    def __init__(self, num_features, num_layers=8, num_heads=8, d_ff=1024, norm=True, nonlinear='relu', dropout=1e-1, causal=False, norm_first=False, eps=EPS):
        super().__init__()

        self.num_features = num_features

        if isinstance(norm, int):
            if norm:
                norm_name = 'cLN' if causal else 'gLN'
                layer_norm = LayerNormWrapper(norm_name, num_features, causal=False, batch_first=False, eps=eps)
            else:
                layer_norm = None
        else:
            norm_name = norm
            layer_norm = LayerNormWrapper(norm_name, num_features, causal=False, batch_first=False, eps=eps)

        self.positional_encoding = PositionalEncoding(num_features, batch_first=False)
        encoder_layer = nn.TransformerEncoderLayer(num_features, num_heads, d_ff, dropout=dropout, activation=nonlinear, layer_norm_eps=eps, batch_first=False, norm_first=norm_first)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, norm=layer_norm)

    def forward(self, input):
        """
        Args:
            input (batch_size, num_features, S, chunk_size)
        Returns:
            output (batch_size, num_features, S, chunk_size)
        """
        num_features = self.num_features
        batch_size, _, S, chunk_size = input.size()

        residual = input
        x = input.permute(2, 0, 3, 1).contiguous() # (batch_size, num_features, S, chunk_size) -> (S, batch_size, chunk_size, num_features)
        x = x.view(S, batch_size * chunk_size, num_features) # (S, batch_size, chunk_size, num_features) -> (S, batch_size * chunk_size, num_features)
        embedding = self.positional_encoding(x)
        x = x + embedding
        x = self.transformer(x) # (S, batch_size*chunk_size, num_features)
        x = x.view(S, batch_size, chunk_size, num_features) # (S, batch_size * chunk_size, num_features) -> (S, batch_size, chunk_size, num_features)
        x = x.permute(1, 3, 0, 2) # (S, batch_size, chunk_size, num_features) -> (batch_size, num_features, S, chunk_size)
        output = x + residual

        return output

class LayerNormWrapper(nn.Module):
    def __init__(self, norm_name, num_features, causal=False, batch_first=False, eps=EPS):
        super().__init__()

        self.batch_first = batch_first

        if norm_name in ['BN', 'batch', 'batch_norm']:
            kwargs = {
                'n_dims': 1
            }
        else:
            kwargs = {}

        self.norm1d = choose_layer_norm(norm_name, num_features, causal=causal, eps=eps, **kwargs)

    def forward(self, input):
        """
        Args:
            input: (T, batch_size, num_features) or (batch_size, T, num_features) if batch_first
        Returns:
            output: (T, batch_size, num_features) or (batch_size, T, num_features) if batch_first
        """
        if self.batch_first:
            input = input.permute(0, 2, 1).contiguous()
        else:
            input = input.permute(1, 2, 0).contiguous()

        output = self.norm1d(input)

        if self.batch_first:
            output = output.permute(0, 2, 1).contiguous()
        else:
            output = output.permute(2, 0, 1).contiguous()

        return output

def _test_intra_transformer():
    num_features = 16
    input = torch.randn((4, num_features, 5, 8))

    model = IntraTransformer(num_features)
    output = model(input)

    print(input.size(), output.size())

def _test_inter_transformer():
    num_features = 16
    input = torch.randn((4, num_features, 5, 8))

    model = InterTransformer(num_features)
    output = model(input)

    print(input.size(), output.size())

def _test_sepformer_block():
    d_model, d_ff = 32, 8
    input = torch.randn((4, d_model, 5, 8))

    model = SepFormerBlock(d_intra=d_model, d_inter=d_model, d_ff_intra=d_ff, d_ff_inter=d_ff)
    output = model(input)

    print(input.size(), output.size())

def _test_sepformer_backbone():
    d_model, d_ff = 32, 8
    input = torch.randn((4, d_model, 5, 8))

    model = SepFormerBackbone(d_intra=d_model, d_inter=d_model, d_ff_intra=d_ff, d_ff_inter=d_ff)
    output = model(input)

    print(input.size(), output.size())

def _test_separator():
    batch_size = 4
    num_features, bottleneck_channels = 3, 32
    d_ff = 8
    T = 128
    input = torch.randn((batch_size, num_features, T))

    model = Separator(num_features, bottleneck_channels, d_ff_intra=d_ff, d_ff_inter=d_ff)
    output = model(input)

    print(input.size(), output.size())

def _test_sepformer():
    kernel_size = 16
    n_basis = 32
    batch_size = 4

    sep_chunk_size, sep_hop_size = 16, 8
    sep_bottleneck_channels = 32
    d_ff = 8
    T = 1024

    input = torch.randn((batch_size, 1, T))

    model = SepFormer(
        n_basis, kernel_size=kernel_size,
        enc_basis='trainable', dec_basis='trainable', enc_nonlinear='relu',
        sep_chunk_size=sep_chunk_size, sep_hop_size=sep_hop_size,
        sep_bottleneck_channels=sep_bottleneck_channels,
        sep_d_ff_intra=d_ff, sep_d_ff_inter=d_ff
    )
    output = model(input)

    print(input.size(), output.size())

if __name__ == '__main__':
    print("="*10, "IntraTransformer", "="*10)
    _test_intra_transformer()
    print()

    print("="*10, "InterTransformer", "="*10)
    _test_inter_transformer()
    print()

    print("="*10, "SepFormerBlock", "="*10)
    _test_sepformer_block()
    print()

    print("="*10, "SepFormerBackbone", "="*10)
    _test_sepformer_backbone()
    print()

    print("="*10, "Separator", "="*10)
    _test_separator()
    print()

    print("="*10, "SepFormer", "="*10)
    _test_sepformer()
