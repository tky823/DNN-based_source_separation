import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.filterbank import choose_filterbank
from utils.model import choose_rnn, choose_nonlinear
from utils.tasnet import choose_layer_norm
from models.gtu import GTU1d
from models.dprnn_tasnet import Segment1d, OverlapAdd1d

EPS = 1e-12

class DPTNet(nn.Module):
    """
        Dual-path transformer based network
    """
    pretrained_model_ids = {
        "wsj0-mix": {
            8000: {
                2: "1QJnJEK8aed7_ED07jD7buyGb37giEDUx",
                3: "1Rfb_vS8r2_Oqpg_zAV9y4WMzv106yrSP"
            },
            16000: {
                2: "", # TODO
                3: "" # TODO
            }
        }
    }
    def __init__(
        self,
        n_basis, kernel_size, stride=None,
        enc_basis=None, dec_basis=None,
        sep_bottleneck_channels=64, sep_hidden_channels=256,
        sep_chunk_size=100, sep_hop_size=None, sep_num_blocks=6,
        sep_num_heads=4, sep_norm=True, sep_nonlinear='relu', sep_dropout=0,
        mask_nonlinear='relu',
        causal=False,
        n_sources=2,
        eps=EPS,
        **kwargs
    ):
        super().__init__()

        if stride is None:
            stride = kernel_size // 2

        if sep_hop_size is None:
            sep_hop_size = sep_chunk_size // 2

        assert kernel_size % stride == 0, "kernel_size is expected divisible by stride"
        assert n_basis % sep_num_heads == 0, "n_basis must be divisible by sep_num_heads"

        # Encoder-decoder
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
        self.sep_bottleneck_channels, self.sep_hidden_channels = sep_bottleneck_channels, sep_hidden_channels
        self.sep_chunk_size, self.sep_hop_size = sep_chunk_size, sep_hop_size
        self.sep_num_blocks = sep_num_blocks
        self.sep_num_heads = sep_num_heads
        self.sep_norm = sep_norm
        self.sep_nonlinear = sep_nonlinear
        self.sep_dropout = sep_dropout

        self.causal = causal
        self.mask_nonlinear = mask_nonlinear

        self.n_sources = n_sources
        self.eps = eps

        # Network configuration
        encoder, decoder = choose_filterbank(n_basis, kernel_size=kernel_size, stride=stride, enc_basis=enc_basis, dec_basis=dec_basis, **kwargs)

        self.encoder = encoder
        self.separator = Separator(
            n_basis, bottleneck_channels=sep_bottleneck_channels, hidden_channels=sep_hidden_channels,
            chunk_size=sep_chunk_size, hop_size=sep_hop_size, num_blocks=sep_num_blocks,
            num_heads=sep_num_heads, norm=sep_norm, nonlinear=sep_nonlinear, dropout=sep_dropout,
            mask_nonlinear=mask_nonlinear,
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

        batch_size, C_in, T = input.size()

        assert C_in == 1, "input.size() is expected (?, 1, ?), but given {}".format(input.size())

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
        x_hat = x_hat.view(batch_size, n_sources, -1)
        output = F.pad(x_hat, (-padding_left, -padding_right))

        return output, latent

    def get_config(self):
        config = {
            'n_basis': self.n_basis,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'enc_basis': self.enc_basis,
            'dec_basis': self.dec_basis,
            'enc_nonlinear': self.enc_nonlinear,
            'window_fn': self.window_fn,
            'enc_onesided': self.enc_onesided,
            'enc_return_complex': self.enc_return_complex,
            'sep_hidden_channels': self.sep_hidden_channels,
            'sep_bottleneck_channels': self.sep_bottleneck_channels,
            'sep_chunk_size': self.sep_chunk_size,
            'sep_hop_size': self.sep_hop_size,
            'sep_num_blocks': self.sep_num_blocks,
            'sep_num_heads': self.sep_num_heads,
            'sep_norm': self.sep_norm,
            'sep_nonlinear': self.sep_nonlinear,
            'sep_dropout': self.sep_dropout,
            'mask_nonlinear': self.mask_nonlinear,
            'causal': self.causal,
            'n_sources': self.n_sources,
            'eps': self.eps
        }

        return config

    @classmethod
    def build_model(cls, model_path, load_state_dict=False):
        config = torch.load(model_path, map_location=lambda storage, loc: storage)

        n_basis = config.get('n_bases') or config['n_basis']
        kernel_size, stride = config['kernel_size'], config['stride']
        enc_basis, dec_basis = config.get('enc_bases') or config['enc_basis'], config.get('dec_bases') or config['dec_basis']
        enc_nonlinear = config['enc_nonlinear']
        enc_onesided, enc_return_complex = config.get('enc_onesided') or None, config.get('enc_return_complex') or None
        window_fn = config['window_fn']

        sep_hidden_channels, sep_bottleneck_channels = config['sep_hidden_channels'], config['sep_bottleneck_channels']
        sep_chunk_size, sep_hop_size = config['sep_chunk_size'], config['sep_hop_size']
        sep_num_blocks = config['sep_num_blocks']
        sep_num_heads = config['sep_num_heads']
        sep_norm, sep_nonlinear, sep_dropout = config['sep_norm'], config['sep_nonlinear'], config['sep_dropout']

        sep_nonlinear, sep_norm = config['sep_nonlinear'], config['sep_norm']
        mask_nonlinear = config['mask_nonlinear']

        causal = config['causal']
        n_sources = config['n_sources']

        eps = config['eps']

        model = cls(
            n_basis, kernel_size, stride=stride,
            enc_basis=enc_basis, dec_basis=dec_basis, enc_nonlinear=enc_nonlinear,
            window_fn=window_fn, enc_onesided=enc_onesided, enc_return_complex=enc_return_complex,
            sep_bottleneck_channels=sep_bottleneck_channels, sep_hidden_channels=sep_hidden_channels,
            sep_chunk_size=sep_chunk_size, sep_hop_size=sep_hop_size, sep_num_blocks=sep_num_blocks,
            sep_num_heads=sep_num_heads,
            sep_norm=sep_norm, sep_nonlinear=sep_nonlinear, sep_dropout=sep_dropout,
            mask_nonlinear=mask_nonlinear,
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
        num_features, bottleneck_channels=32, hidden_channels=128,
        chunk_size=100, hop_size=None, num_blocks=6,
        num_heads=4,
        norm=True, nonlinear='relu', dropout=0,
        mask_nonlinear='relu',
        causal=True,
        n_sources=2,
        eps=EPS
    ):
        super().__init__()

        if hop_size is None:
            hop_size = chunk_size // 2

        self.num_features, self.n_sources = num_features, n_sources
        self.chunk_size, self.hop_size = chunk_size, hop_size

        self.bottleneck_conv1d = nn.Conv1d(num_features, bottleneck_channels, kernel_size=1, stride=1)
        self.segment1d = Segment1d(chunk_size, hop_size)

        norm_name = 'cLN' if causal else 'gLN'
        self.norm2d = choose_layer_norm(norm_name, bottleneck_channels, causal=causal, eps=eps)

        self.dptransformer = DualPathTransformer(
            bottleneck_channels, hidden_channels,
            num_blocks=num_blocks, num_heads=num_heads,
            norm=norm, nonlinear=nonlinear, dropout=dropout,
            causal=causal, eps=eps
        )
        self.overlap_add1d = OverlapAdd1d(chunk_size, hop_size)
        self.prelu = nn.PReLU()
        self.map = nn.Conv1d(bottleneck_channels, n_sources*num_features, kernel_size=1, stride=1)
        self.gtu = GTU1d(num_features, num_features, kernel_size=1, stride=1)

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

        x = self.bottleneck_conv1d(input)
        x = F.pad(x, (padding_left, padding_right))
        x = self.segment1d(x) # -> (batch_size, C, S, chunk_size)
        x = self.norm2d(x)
        x = self.dptransformer(x)
        x = self.overlap_add1d(x)
        x = F.pad(x, (-padding_left, -padding_right))
        x = self.prelu(x) # -> (batch_size, C, n_frames)
        x = self.map(x) # -> (batch_size, n_sources*C, n_frames)
        x = x.view(batch_size*n_sources, num_features, n_frames) # -> (batch_size*n_sources, num_features, n_frames)
        x = self.gtu(x) # -> (batch_size*n_sources, num_features, n_frames)
        x = self.mask_nonlinear(x) # -> (batch_size*n_sources, num_features, n_frames)
        output = x.view(batch_size, n_sources, num_features, n_frames)

        return output

class DualPathTransformer(nn.Module):
    def __init__(self, num_features, hidden_channels, num_blocks=6, num_heads=4, norm=True, nonlinear='relu', dropout=0, causal=False, eps=EPS):
        super().__init__()

        # Network confguration
        net = []

        for _ in range(num_blocks):
            net.append(DualPathTransformerBlock(num_features, hidden_channels, num_heads=num_heads, norm=norm, nonlinear=nonlinear, dropout=dropout, causal=causal, eps=eps))

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

class DualPathTransformerBlock(nn.Module):
    def __init__(self, num_features, hidden_channels, num_heads=4, norm=True, nonlinear='relu', dropout=0, causal=False, eps=EPS):
        super().__init__()

        self.intra_chunk_block = IntraChunkTransformer(
            num_features, hidden_channels, num_heads=num_heads,
            norm=norm, nonlinear=nonlinear, dropout=dropout,
            eps=eps
        )
        self.inter_chunk_block = InterChunkTransformer(
            num_features, hidden_channels, num_heads=num_heads,
            norm=norm, nonlinear=nonlinear, dropout=dropout,
            causal=causal,
            eps=eps
        )

    def forward(self, input):
        """
        Args:
            input (batch_size, num_features, S, chunk_size)
        Returns:
            output (batch_size, num_features, S, chunk_size)
        """
        x = self.intra_chunk_block(input)
        output = self.inter_chunk_block(x)

        return output

class IntraChunkTransformer(nn.Module):
    def __init__(self, num_features, hidden_channels, num_heads=4, norm=True, nonlinear='relu', dropout=0, eps=EPS):
        super().__init__()

        self.num_features = num_features

        self.transformer = ImprovedTransformer(
            num_features, hidden_channels, num_heads=num_heads,
            norm=norm, nonlinear=nonlinear, dropout=dropout,
            causal=False,
            eps=eps
        )

    def forward(self, input):
        """
        Args:
            input (batch_size, num_features, S, chunk_size)
        Returns:
            output (batch_size, num_features, S, chunk_size)
        """
        num_features = self.num_features
        batch_size, _, S, chunk_size = input.size()

        x = input.permute(3, 0, 2, 1).contiguous() # (batch_size, num_features, S, chunk_size) -> (chunk_size, batch_size, S, num_features)
        x = x.view(chunk_size, batch_size*S, num_features) # (chunk_size, batch_size, S, num_features) -> (chunk_size, batch_size*S, num_features)
        x = self.transformer(x) # -> (chunk_size, batch_size*S, num_features)
        x = x.view(chunk_size, batch_size, S, num_features) # (chunk_size, batch_size*S, num_features) -> (chunk_size, batch_size, S, num_features)
        output = x.permute(1, 3, 2, 0) # (chunk_size, batch_size, S, num_features) -> (batch_size, num_features, S, chunk_size)

        return output

class InterChunkTransformer(nn.Module):
    def __init__(self, num_features, hidden_channels, num_heads=4, causal=False, norm=True, nonlinear='relu', dropout=0, eps=EPS):
        super().__init__()

        self.num_features = num_features

        self.transformer = ImprovedTransformer(
            num_features, hidden_channels, num_heads=num_heads,
            norm=norm, nonlinear=nonlinear, dropout=dropout,
            causal=causal,
            eps=eps
        )

    def forward(self, input):
        """
        Args:
            input (batch_size, num_features, S, chunk_size)
        Returns:
            output (batch_size, num_features, S, chunk_size)
        """
        num_features = self.num_features
        batch_size, _, S, chunk_size = input.size()

        x = input.permute(2, 0, 3, 1).contiguous() # (batch_size, num_features, S, chunk_size) -> (S, batch_size, chunk_size, num_features)
        x = x.view(S, batch_size*chunk_size, num_features) # (S, batch_size, chunk_size, num_features) -> (S, batch_size*chunk_size, num_features)
        x = self.transformer(x) # -> (S, batch_size*chunk_size, num_features)
        x = x.view(S, batch_size, chunk_size, num_features) # (S, batch_size*chunk_size, num_features) -> (S, batch_size, chunk_size, num_features)
        output = x.permute(1, 3, 0, 2) # (S, batch_size, chunk_size, num_features) -> (batch_size, num_features, S, chunk_size)

        return output

class ImprovedTransformer(nn.Module):
    def __init__(self, num_features, hidden_channels, num_heads=4, norm=True, nonlinear='relu', dropout=0, causal=False, eps=EPS):
        super().__init__()

        self.multihead_attn_block = MultiheadAttentionBlock(num_features, num_heads, norm=norm, dropout=dropout, causal=causal, eps=eps)
        self.subnet = FeedForwardBlock(num_features, hidden_channels, norm=norm, nonlinear=nonlinear, causal=causal, eps=eps)

    def forward(self, input):
        """
        Args:
            input (T, batch_size, num_features)
        Returns:
            output (T, batch_size, num_features)
        """
        x = self.multihead_attn_block(input)
        output = self.subnet(x)

        return output

class MultiheadAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, norm=True, dropout=0, causal=False, eps=EPS):
        super().__init__()

        if dropout == 0:
            self.dropout = False
        else:
            self.dropout = True

        self.norm = norm

        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)

        if self.dropout:
            self.dropout1d = nn.Dropout(p=dropout)

        if self.norm:
            norm_name = 'cLN' if causal else 'gLN'
            self.norm1d = choose_layer_norm(norm_name, embed_dim, causal=causal, eps=eps)

    def forward(self, input):
        """
        Args:
            input (T, batch_size, embed_dim)
        Returns:
            output (T, batch_size, embed_dim)
        """
        x = input # (T, batch_size, embed_dim)

        residual = x
        x, _ = self.multihead_attn(x, x, x) # (T_tgt, batch_size, embed_dim), (batch_size, T_tgt, T_src), where T_tgt = T_src = T
        x = x + residual

        if self.dropout:
            x = self.dropout1d(x)

        if self.norm:
            x = x.permute(1, 2, 0) # (batch_size, embed_dim, T)
            x = self.norm1d(x) # (batch_size, embed_dim, T)
            x = x.permute(2, 0, 1).contiguous() # (batch_size, embed_dim, T) -> (T, batch_size, embed_dim)

        output = x

        return output

class FeedForwardBlock(nn.Module):
    def __init__(self, num_features, hidden_channels, norm=True, nonlinear='relu', causal=False, eps=EPS):
        super().__init__()

        if causal:
            bidirectional = False
            num_directions = 1 # uni-direction
        else:
            bidirectional = True
            num_directions = 2 # bi-direction

        self.norm = norm

        self.rnn = choose_rnn('lstm', input_size=num_features, hidden_size=hidden_channels, batch_first=False, bidirectional=bidirectional) # TODO: rnn_type
        self.nonlinear1d = choose_nonlinear(nonlinear)
        self.fc = nn.Linear(num_directions*hidden_channels, num_features)

        if self.norm:
            norm_name = 'cLN' if causal else 'gLN'
            self.norm1d = choose_layer_norm(norm_name, num_features, causal=causal, eps=eps)

    def forward(self, input):
        """
        Args:
            input (T, batch_size, num_features)
        Returns:
            output (T, batch_size, num_features)
        """
        x = input # (T, batch_size, num_features)

        self.rnn.flatten_parameters()

        residual = x
        x, _ = self.rnn(x) # (T, batch_size, num_features) -> (T, batch_size, num_directions*hidden_channels)
        x = self.nonlinear1d(x) # -> (T, batch_size, num_directions*hidden_channels)
        x = self.fc(x) # (T, batch_size, num_directions*hidden_channels) -> (T, batch_size, num_features)
        x = x + residual

        if self.norm:
            x = x.permute(1, 2, 0) # (T, batch_size, num_features) -> (batch_size, num_features, T)
            x = self.norm1d(x) # (batch_size, num_features, T)
            x = x.permute(2, 0, 1).contiguous() # (batch_size, num_features, T) -> (T, batch_size, num_features)

        output = x

        return output

def _test_multihead_attn_block():
    batch_size = 2
    T = 10
    embed_dim = 8
    num_heads = 4
    input = torch.randn((T, batch_size, embed_dim), dtype=torch.float)

    print('-'*10, "Non causal & No dropout", '-'*10)
    causal = False
    dropout = 0

    model = MultiheadAttentionBlock(embed_dim, num_heads=num_heads, dropout=dropout, causal=causal)
    print(model)

    output = model(input)
    print(input.size(), output.size())
    print()

    print('-'*10, "Causal & Dropout (p=0.3)", '-'*10)
    causal = True
    dropout = 0.3

    model = MultiheadAttentionBlock(embed_dim, num_heads=num_heads, dropout=dropout, causal=causal)
    print(model)

    output = model(input)
    print(input.size(), output.size())

def _test_feedforward_block():
    batch_size = 2
    T = 10
    num_features, hidden_channels = 12, 10

    input = torch.randn((T, batch_size, num_features), dtype=torch.float)

    print('-'*10, "Causal", '-'*10)
    causal = True
    nonlinear = 'relu'

    model = FeedForwardBlock(num_features, hidden_channels, nonlinear=nonlinear, causal=causal)
    print(model)

    output = model(input)
    print(input.size(), output.size())

def _test_improved_transformer():
    batch_size = 2
    T = 10
    num_features, hidden_channels = 12, 10
    num_heads = 4

    input = torch.randn((T, batch_size, num_features), dtype=torch.float)

    print('-'*10, "Non causal", '-'*10)
    causal = False

    model = ImprovedTransformer(num_features, hidden_channels, num_heads=num_heads, causal=causal)
    print(model)

    output = model(input)
    print(input.size(), output.size())

def _test_transformer_block():
    batch_size = 2
    num_features, hidden_channels = 12, 8
    S, chunk_size = 10, 5 # global length and local length
    num_heads = 3
    input = torch.randn((batch_size, num_features, S, chunk_size), dtype=torch.float)

    print('-'*10, "transformer block for intra chunk", '-'*10)
    model = IntraChunkTransformer(num_features, hidden_channels=hidden_channels, num_heads=num_heads)
    print(model)

    output = model(input)
    print(input.size(), output.size())
    print()

    print('-'*10, "transformer block for inter chunk", '-'*10)
    causal = True
    model = InterChunkTransformer(num_features, hidden_channels=hidden_channels, num_heads=num_heads, causal=causal)
    print(model)

    output = model(input)
    print(input.size(), output.size())

def _test_dptransformer():
    batch_size = 2
    num_features, hidden_channels = 12, 8
    S, chunk_size = 10, 5 # global length and local length
    num_blocks = 6
    num_heads = 3
    input = torch.randn((batch_size, num_features, S, chunk_size), dtype=torch.float)
    causal = True

    model = DualPathTransformer(num_features, hidden_channels, num_blocks=num_blocks, num_heads=num_heads, causal=causal)
    print(model)

    output = model(input)
    print(input.size(), output.size())

def _test_separator():
    batch_size = 2
    T_bin = 64
    n_sources = 3

    num_features = 10
    d = 12 # must be divisible by num_heads
    d_ff = 15
    chunk_size = 10 # local chunk length
    num_blocks = 3
    num_heads = 4 # multihead attention in transformer

    input = torch.randn((batch_size, num_features, T_bin), dtype=torch.float)

    causal = False

    separator = Separator(
        num_features, hidden_channels=d_ff, bottleneck_channels=d,
        chunk_size=chunk_size, num_blocks=num_blocks, num_heads=num_heads,
        causal=causal,
        n_sources=n_sources
    )
    print(separator)

    output = separator(input)
    print(input.size(), output.size())

def _test_dptnet():
    batch_size = 2
    T = 64

    # Encoder decoder
    N, L = 12, 8
    enc_basis, dec_basis = 'trainable', 'trainable'
    enc_nonlinear = 'relu'
    
    # Separator
    d = 32 # must be divisible by num_heads
    d_ff = 4 * d # depth of feed-forward network
    K = 10 # local chunk length
    B, h = 3, 4 # number of dual path transformer processing block, and multihead attention in transformer
    mask_nonlinear = 'relu'
    n_sources = 2

    input = torch.randn((batch_size, 1, T), dtype=torch.float)

    causal = False

    model = DPTNet(
        N, L, enc_basis=enc_basis, dec_basis=dec_basis, enc_nonlinear=enc_nonlinear,
        sep_bottleneck_channels=d, sep_hidden_channels=d_ff,
        sep_chunk_size=K, sep_num_blocks=B, sep_num_heads=h,
        mask_nonlinear=mask_nonlinear,
        causal=causal,
        n_sources=n_sources
    )
    print(model)

    output = model(input)
    print("# Parameters: {}".format(model.num_parameters))
    print(input.size(), output.size())

def _test_dptnet_paper():
    batch_size = 2
    T = 64

    # Encoder decoder
    N, L = 64, 2
    enc_basis, dec_basis = 'trainable', 'trainable'
    enc_nonlinear = 'relu'

    # Separator
    d = 32
    d_ff = 4 * d # depth of feed-forward network
    K = 10 # local chunk length
    B, h = 6, 4 # number of dual path transformer processing block, and multihead attention in transformer

    mask_nonlinear = 'relu'
    n_sources = 2

    input = torch.randn((batch_size, 1, T), dtype=torch.float)

    causal = False

    model = DPTNet(
        N, L, enc_basis=enc_basis, dec_basis=dec_basis, enc_nonlinear=enc_nonlinear,
        sep_bottleneck_channels=N, sep_hidden_channels=d_ff,
        sep_chunk_size=K, sep_num_blocks=B, sep_num_heads=h,
        mask_nonlinear=mask_nonlinear,
        causal=causal,
        n_sources=n_sources
    )
    print(model)

    output = model(input)
    print("# Parameters: {}".format(model.num_parameters))
    print(input.size(), output.size())

if __name__ == '__main__':
    print('='*10, "Multihead attention block", '='*10)
    _test_multihead_attn_block()
    print()

    print('='*10, "feed-forward block", '='*10)
    _test_feedforward_block()
    print()

    print('='*10, "improved transformer", '='*10)
    _test_improved_transformer()
    print()

    print('='*10, "transformer block", '='*10)
    _test_transformer_block()
    print()

    print('='*10, "Dual path transformer network", '='*10)
    _test_dptransformer()
    print()

    print('='*10, "Separator based on dual path transformer network", '='*10)
    _test_separator()
    print()

    print('='*10, "Dual path transformer network", '='*10)
    _test_dptnet()
    print()

    print('='*10, "Dual path transformer network (same configuration in the paper)", '='*10)
    _test_dptnet_paper()
    print()