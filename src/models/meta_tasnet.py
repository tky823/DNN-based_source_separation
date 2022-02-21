import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-12

class MetaTasNet(nn.Module):
    def __init__(self,
        n_bases, kernel_size, stride=None,
        enc_fft_size=None, enc_hop_size=None, enc_compression_rate=4, num_filters=6, n_mels=256,
        sep_hidden_channels=256, sep_bottleneck_channels=128, sep_skip_channels=128,
        sep_kernel_size=3, sep_num_blocks=3, sep_num_layers=8,
        dilated=True, separable=True, dropout=0.0,
        sep_nonlinear='prelu', mask_nonlinear='sigmoid',
        causal=False,
        conv_name='generated', norm_name='generated',
        num_stages=3, n_sources=2,
        eps=EPS, **kwargs
    ):
        super().__init__()

        self.num_stages = num_stages

        if stride is None:
            stride = kernel_size
        
        # Encoder & Decoder configuration
        self.n_bases = n_bases
        self.kernel_size, self.stride = kernel_size, stride
        self.enc_fft_size, self.enc_hop_size = enc_hop_size, enc_hop_size
        self.enc_compression_rate = enc_compression_rate
        self.num_filters, self.n_mels = num_filters, n_mels

        # Separator configuration
        self.conv_name, self.norm_name = conv_name, norm_name

        self.sep_hidden_channels, self.sep_bottleneck_channels, self.sep_skip_channels = sep_hidden_channels, sep_bottleneck_channels, sep_skip_channels
        self.sep_kernel_size = sep_kernel_size
        self.sep_num_blocks, self.sep_num_layers = sep_num_blocks, sep_num_layers
        
        self.dilated, self.separable, self.causal = dilated, separable, causal
        self.sep_nonlinear = sep_nonlinear
        self.mask_nonlinear = mask_nonlinear

        self._additional_keys = []

        for key in kwargs:
            if not isinstance(kwargs[key], nn.Module):
                setattr(self, key, kwargs[key])
                self._additional_keys.append(key)

        # Others
        self.dropout = dropout
        self.n_sources = n_sources
        self.eps = eps

        net = []
        sep_in_channels = 0

        for idx in range(num_stages):
            scale = 2**idx
            sep_in_channels += scale*n_bases
            
            backbone = MetaTasNetBackbone(
                scale*n_bases, scale*kernel_size, stride=scale*stride,
                enc_fft_size=scale*enc_fft_size, enc_hop_size=scale*enc_hop_size, enc_compression_rate=enc_compression_rate, num_filters=num_filters, n_mels=n_mels,
                sep_in_channels=sep_in_channels, sep_hidden_channels=sep_hidden_channels, sep_bottleneck_channels=sep_bottleneck_channels, sep_skip_channels=sep_skip_channels,
                sep_kernel_size=sep_kernel_size, sep_num_blocks=sep_num_blocks, sep_num_layers=sep_num_layers,
                dilated=dilated, separable=separable, dropout=dropout,
                sep_nonlinear=sep_nonlinear, mask_nonlinear=mask_nonlinear,
                causal=causal,
                conv_name=conv_name, norm_name=norm_name,
                n_sources=n_sources,
                eps=eps, **kwargs
            )
            net.append(backbone)
            sep_in_channels = scale*n_bases
        
        self.net = nn.ModuleList(net)

    def forward(self, input, masking=True, max_stage=None):
        latent = None
        outputs = []

        if max_stage is None:
            max_stage = len(input)

        for idx in range(max_stage):
            output, latent = self.net[idx].extract_latent(input[idx], latent=latent, masking=masking)
            outputs.append(output)

        return outputs

    def extract_latent(self, input, masking=True, max_stage=None):
        """
        Args:
            input <torch.Tensor>: (batch_size, n_sources, num_stages, n_bases, n_frames)
            masking <bool>: Apply mask or not
        Returns:
            outputs: List of outputs
            latents: List of latents
        """
        latent = None
        outputs, latents = [], []

        if max_stage is None:
            max_stage = len(input)
        
        for idx in range(max_stage):
            output, latent = self.net[idx].extract_latent(input[idx], latent=latent, masking=masking)
            outputs.append(output)
            latents.append(latent)

        return outputs, latents
    
    def forward_separators(self, input, max_stage=None):
        """
        Forward separators (masking modules)
        """
        warnings.warn("in progress", UserWarning)
        
        outputs = []

        if max_stage is None:
            max_stage = len(input)
        
        for idx in range(max_stage):
            latent = input[idx]
            mask = self.net[idx].forward_separator(latent)
            outputs.append(mask)
        
        return outputs

    def forward_decoders(self, input, mask=None, max_stage=None):
        warnings.warn("in progress", UserWarning)
        
        outputs = []

        if max_stage is None:
            max_stage = len(input)
        
        for idx in range(max_stage):
            if mask is None:
                latent = input[idx]
            else:
                latent = mask[idx] * input[idx]
            
            x_hat = self.net[idx].forward_decoder(latent)
            outputs.append(x_hat)
        
        return outputs
    
    def get_package(self):
        return self.get_config()
    
    def get_config(self):
        config = {
            'n_bases': self.n_bases,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'enc_fft_size': self.enc_fft_size,
            'enc_hop_size': self.enc_hop_size,
            'enc_compression_rate': self.enc_compression_rate,
            'num_filters': self.num_filters,
            'n_mels': self.n_mels,
            'sep_hidden_channels': self.sep_hidden_channels,
            'sep_bottleneck_channels': self.sep_bottleneck_channels,
            'sep_skip_channels': self.sep_skip_channels,
            'sep_kernel_size': self.sep_kernel_size,
            'sep_num_blocks': self.sep_num_blocks,
            'sep_num_layers': self.sep_num_layers,
            'dilated': self.dilated,
            'separable': self.separable,
            'dropout': self.dropout,
            'sep_nonlinear': self.sep_nonlinear,
            'mask_nonlinear': self.mask_nonlinear,
            'causal': self.causal,
            'conv_name': self.conv_name,
            'norm_name': self.norm_name,
            'num_stages': self.num_stages,
            'n_sources': self.n_sources,
            'eps': self.eps
        }

        kwargs = {}

        for key in self._additional_keys:
            kwargs[key] = getattr(self, key)
        
        config['kwargs'] = kwargs
        
        return config
    
    @classmethod
    def build_model(cls, model_path):
        config = torch.load(model_path, map_location=lambda storage, loc: storage)

        n_bases = config['n_bases']
        kernel_size, stride = config['kernel_size'], config['stride']
        enc_fft_size, enc_hop_size = config['enc_fft_size'], config['enc_hop_size']
        enc_compression_rate = config['enc_compression_rate']
        num_filters, n_mels = config['num_filters'], config['n_mels']

        sep_hidden_channels, sep_bottleneck_channels, sep_skip_channels = config['sep_hidden_channels'], config['sep_bottleneck_channels'], config['sep_skip_channels']
        sep_kernel_size = config['sep_kernel_size']
        sep_num_blocks, sep_num_layers = config['sep_num_blocks'], config['sep_num_layers']

        dilated, separable, causal = config['dilated'], config['separable'], config['causal']
        dropout = config['dropout']
        sep_nonlinear, mask_nonlinear = config['sep_nonlinear'], config['mask_nonlinear']

        conv_name, norm_name = config['conv_name'], config['norm_name']
        
        num_stages = config['num_stages']
        n_sources = config['n_sources']

        eps = config['eps']

        kwargs = config['kwargs']
        
        model = cls(
            n_bases, kernel_size, stride=stride,
            enc_fft_size=enc_fft_size, enc_hop_size=enc_hop_size, enc_compression_rate=enc_compression_rate, num_filters=num_filters, n_mels=n_mels,
            sep_hidden_channels=sep_hidden_channels, sep_bottleneck_channels=sep_bottleneck_channels, sep_skip_channels=sep_skip_channels,
            sep_kernel_size=sep_kernel_size, sep_num_blocks=sep_num_blocks, sep_num_layers=sep_num_layers,
            dilated=dilated, separable=separable, dropout=dropout,
            sep_nonlinear=sep_nonlinear, mask_nonlinear=mask_nonlinear,
            causal=causal,
            conv_name=conv_name, norm_name=norm_name,
            num_stages=num_stages, n_sources=n_sources,
            eps=eps, **kwargs
        )
        
        return model

    @property
    def num_parameters(self):
        _num_parameters = 0
        
        for p in self.parameters():
            if p.requires_grad:
                _num_parameters += p.numel()
                
        return _num_parameters

class MetaTasNetBackbone(nn.Module):
    def __init__(self,
        n_bases, kernel_size, stride=None,
        enc_fft_size=None, enc_hop_size=None, enc_compression_rate=4, num_filters=6, n_mels=256,
        sep_in_channels=None, sep_hidden_channels=256, sep_bottleneck_channels=128, sep_skip_channels=128,
        sep_kernel_size=3, sep_num_blocks=3, sep_num_layers=8,
        dilated=True, separable=True, dropout=0.0,
        sep_nonlinear='prelu', mask_nonlinear='sigmoid',
        causal=False,
        conv_name='generated', norm_name='generated',
        n_sources=2,
        eps=EPS, **kwargs
    ):
        """
        Args:
            n_bases
            kernel_size
            stride
            enc_fft_size
            enc_hop_size
            enc_compression_rate
            num_filters
            n_mels
            sep_in_channels
            sep_hidden_channels
            sep_bottleneck_channels
            sep_skip_channels
            sep_kernel_size
            sep_num_blocks
            sep_num_layers
            dilated
            separable
            dropout
            sep_nonlinear
            mask_nonlinear
            causal,
            conv_name
            norm_name
            n_sources
            eps
        """
        super().__init__()
        
        # Encoder & Decoder configuration
        self.n_bases = n_bases
        self.kernel_size, self.stride = kernel_size, stride
        self.enc_fft_size, self.enc_hop_size = enc_hop_size, enc_hop_size
        self.enc_compression_rate = enc_compression_rate
        self.num_filters, self.n_mels = num_filters, n_mels

        # Separator configuration
        self.conv_name, self.norm_name = conv_name, norm_name

        self.sep_in_channels = sep_in_channels
        self.sep_hidden_channels, self.sep_bottleneck_channels, self.sep_skip_channels = sep_hidden_channels, sep_bottleneck_channels, sep_skip_channels
        self.sep_kernel_size = sep_kernel_size
        self.sep_num_blocks, self.sep_num_layers = sep_num_blocks, sep_num_layers
        
        self.dilated, self.separable, self.causal = dilated, separable, causal
        self.sep_nonlinear = sep_nonlinear
        self.mask_nonlinear = mask_nonlinear

        # Others
        self.dropout = dropout
        self.n_sources = n_sources
        self.eps = eps

        self.encoder = Encoder(n_bases, kernel_size, stride=stride, fft_size=enc_fft_size, hop_size=enc_hop_size, n_mels=n_mels, num_filters=num_filters, compression_rate=enc_compression_rate)

        self.dropout2d = nn.Dropout2d(dropout)
        
        if norm_name == 'generated':
            embed_dim = kwargs['embed_dim']
            self.embedding = nn.Embedding(n_sources, embed_dim)
        else:
            self.embedding = None

        if sep_in_channels is None:
            sep_in_channels = n_bases
        
        self.separator = Separator(
            sep_in_channels, n_bases, bottleneck_channels=sep_bottleneck_channels, hidden_channels=sep_hidden_channels, skip_channels=sep_skip_channels,
            kernel_size=sep_kernel_size, num_blocks=sep_num_blocks, num_layers=sep_num_layers, dilated=dilated, separable=separable, causal=causal, nonlinear=sep_nonlinear, mask_nonlinear=mask_nonlinear,
            conv_name=conv_name, norm_name=norm_name,
            n_sources=n_sources,
            eps=eps,
            **kwargs
        )

        self.decoder = Decoder(n_bases, kernel_size, stride=stride, num_filters=num_filters)
    
    def forward(self, input, latent=None, masking=True):
        output, _ = self.extract_latent(input, latent=latent, masking=masking)

        return output
    
    def extract_latent(self, input, latent=None, masking=True):
        """
        Args:
            input: (batch_size, 1, T)
        Returns:
            output: (batch_size, n_sources, T)
            latent: (batch_size, n_sources, num_features, n_frames) or (batch_size, 1, num_features, n_frames) if masking = False
        """
        n_sources = self.n_sources
        n_bases = self.n_bases
        kernel_size, stride = self.kernel_size, self.stride

        n_dims = input.dim()

        if n_dims == 3:
            batch_size, C_in, T = input.size()
            assert C_in == 1, "input.size() is expected (?, 1, ?), but given {}".format(input.size())
        else:
            raise ValueError("Not support {} dimension input".format(n_dims))
        
        padding = kernel_size - stride
        padding_left = padding // 2
        padding_right = padding - padding_left

        input = F.pad(input, (padding_left, padding_right))
        w = self.encoder(input)

        batch_size, num_features, n_frames = w.size()
        w = w.unsqueeze(dim=1) # (batch_size, 1, num_features, n_frames)

        if masking:
            w_repeated = w.repeat(1, n_sources, 1, 1).contiguous() # (batch_size, n_sources, num_features, n_frames)

            if latent is not None:
                w_repeated = torch.cat([w_repeated, latent], dim=2) # (batch_size, n_sources, sep_in_channels, n_frames)
            
            # TODO: dropout2d?
            w_repeated = self.dropout2d(w_repeated) # (batch_size, n_sources, sep_in_channels, n_frames)

            if self.embedding:
                input_source = torch.arange(n_sources).long()
                input_source = input_source.to(w_repeated.device)
                embedding = self.embedding(input_source)
                mask = self.separator(w_repeated, embedding=embedding) # (batch_size, n_sources, n_bases, n_frames)
            else:
                mask = self.separator(w_repeated) # (batch_size, n_sources, n_bases, n_frames)
            
            w_hat = w * mask
            latent = w_hat
            w_hat = w_hat.view(batch_size * n_sources, n_bases, n_frames)

            x_hat = self.decoder(w_hat)
            x_hat = x_hat.view(batch_size, n_sources, -1) # (batch_size, n_sources, T_pad)
        else:
            w = self.dropout2d(w) # (batch_size, 1, num_features, n_frames)
            latent = w # (batch_size, 1, num_features, n_frames)
            w = w.view(batch_size, n_bases, n_frames)

            x_hat = self.decoder(w) # (batch_size, 1, T_pad)

        output = F.pad(x_hat, (-padding_left, -padding_right))
    
        return output, latent

    def forward_separator(self, input):
        n_sources = self.n_sources

        if self.embedding:
            input_source = torch.arange(n_sources).long()
            embedding = self.embedding(input_source)
            output = self.separator(input, embedding=embedding) # (batch_size, n_sources, n_bases, n_frames)
        else:
            output = self.separator(input)
        
        return output

    def forward_decoder(self, input):
        output = self.decoder(input)
        return output

    def get_package(self):
        return self.get_package()

    def get_config(self):
        package = {
            'n_bases': self.n_bases,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'enc_fft_size': self.enc_fft_size,
            'enc_hop_size': self.enc_hop_size,
            'enc_compression_rate': self.enc_compression_rate,
            'num_filters': self.num_filters,
            'n_mels': self.n_mels,
            'sep_in_channels': self.sep_in_channels,
            'sep_hidden_channels': self.sep_hidden_channels,
            'sep_bottleneck_channels': self.sep_bottleneck_channels,
            'sep_skip_channels': self.sep_skip_channels,
            'sep_kernel_size': self.sep_kernel_size,
            'sep_num_blocks': self.sep_num_blocks,
            'sep_num_layers': self.sep_num_layers,
            'dilated': self.dilated,
            'separable': self.separable,
            'dropout': self.dropout,
            'sep_nonlinear': self.sep_nonlinear,
            'mask_nonlinear': self.mask_nonlinear,
            'causal': self.causal,
            'conv_name': self.conv_name,
            'norm_name': self.norm_name,
            'n_sources': self.n_sources,
            'eps': self.eps
        }
        
        return package

class Encoder(nn.Module):
    def __init__(self, n_bases, kernel_size, stride=20, fft_size=None, hop_size=None, n_mels=256, num_filters=6, compression_rate=4):

        super().__init__()

        if hop_size is None:
            hop_size = fft_size // 4
        
        self.num_filters = num_filters
        
        self.spectrogram = Spectrogram(fft_size=fft_size, hop_size=hop_size, n_mels=n_mels)

        _out_channels = n_bases // compression_rate
        out_channels = 0
        filters = []

        for idx in range(num_filters):
            _kernel_size = kernel_size * (2**idx)
            filters.append(nn.Conv1d(1, _out_channels, kernel_size=_kernel_size, stride=stride, bias=False, padding=(_kernel_size - stride)//2))
            out_channels += _out_channels
        
        out_channels += n_mels

        self.filters = nn.ModuleList(filters)
        self.nonlinear = nn.ReLU()
        self.postprocess = nn.Sequential(
            nn.Conv1d(out_channels, n_bases, kernel_size=1, stride=1, bias=False),
            nn.ReLU(),
            nn.Conv1d(n_bases, n_bases, kernel_size=1, stride=1, bias=False),
        )

    def forward(self, input):
        num_filters = self.num_filters
        latent = []

        for idx in range(num_filters):
            x = self.filters[idx](input)
            latent.append(x)

        x = torch.cat(latent, dim=1)
        x = self.nonlinear(x)

        batch_size, _, T = input.size()
        input = input.view(-1, T)
        x_spectrogram = self.spectrogram(input, x.size(-1))
        x_spectrogram = x_spectrogram.view(batch_size, *x_spectrogram.size()[-2:])
        x = torch.cat([x, x_spectrogram], dim=1)
        output = self.postprocess(x)

        return output

class Decoder(nn.Module):
    def __init__(self, n_bases, kernel_size, stride=20, num_filters=6):
        super().__init__()

        self.sections = [n_bases // (2**(idx + 1)) for idx in range(num_filters)]
        out_channels = sum(self.sections)

        self.preprocess = nn.Sequential(
            nn.ConvTranspose1d(n_bases, out_channels, kernel_size=1, stride=1, bias=False),
            nn.ReLU()
        )
        
        filters = []

        for idx in range(num_filters):
            _in_channels = n_bases // (2**(idx + 1))
            _kernel_size = kernel_size * (2**idx)
            filters.append(nn.ConvTranspose1d(_in_channels, 1, kernel_size=_kernel_size, stride=stride, bias=False, padding=(_kernel_size - stride)//2))

        self.filters = nn.ModuleList(filters)

    def forward(self, input):
        x = self.preprocess(input)
        x = torch.split(x, self.sections, dim=1)

        output = 0

        for idx in range(len(x)):
            output = output + self.filters[idx](x[idx])
        
        return output

class Separator(nn.Module):
    def __init__(self,
        in_channels, out_channels, bottleneck_channels=128, hidden_channels=256, skip_channels=128,
        kernel_size=3, num_blocks=3, num_layers=8, dilated=True, separable=True, causal=False, nonlinear='prelu', mask_nonlinear='softmax',
        conv_name='generated', norm_name='generated',
        n_sources=2,
        eps=EPS,
        **kwargs
    ):
        super().__init__()
        
        self.in_channels, self.out_channels, self.n_sources = in_channels, out_channels, n_sources

        kwargs_meta = get_kwargs_meta(kwargs)
        
        self.norm1d = choose_layer_norm(norm_name, in_channels, causal=causal, n_sources=n_sources, eps=eps, **kwargs_meta)
        self.bottleneck_conv1d = choose_conv1d(conv_name, in_channels, bottleneck_channels, kernel_size=1, stride=1, n_sources=n_sources, **kwargs_meta)

        self.tcn = TemporalConvNet(
            bottleneck_channels, hidden_channels=hidden_channels, skip_channels=skip_channels, kernel_size=kernel_size, num_blocks=num_blocks, num_layers=num_layers,
            dilated=dilated, separable=separable, causal=causal, nonlinear=nonlinear,
            conv_name=conv_name, norm_name=norm_name, n_sources=n_sources,
            eps=eps,
            **kwargs_meta
        )
        self.prelu = nn.PReLU()
        self.mask_conv1d = choose_conv1d(conv_name, skip_channels, out_channels, kernel_size=1, stride=1, n_sources=n_sources, **kwargs_meta)
        
        if mask_nonlinear == 'sigmoid':
            self.mask_nonlinear = nn.Sigmoid()
        elif mask_nonlinear == 'softmax':
            self.mask_nonlinear = nn.Softmax(dim=1)
        else:
            raise ValueError("Cannot support {}".format(mask_nonlinear))
        
    def forward(self, input, embedding=None):
        """
        Args:
            input (batch_size, n_sources, in_channels, n_frames)
        Returns:
            output (batch_size, n_sources, out_channels, n_frames)
        """
        out_channels, n_sources = self.out_channels, self.n_sources

        batch_size, _, _, n_frames = input.size()
        
        if embedding is not None:
            x = self.norm1d(input, embedding=embedding)
            x = self.bottleneck_conv1d(x, embedding=embedding)
            x = self.tcn(x, embedding=embedding)
            x = self.prelu(x)
            x = self.mask_conv1d(x, embedding=embedding)
        else:
            x = self.norm1d(input)
            x = self.bottleneck_conv1d(x)
            x = self.tcn(x)
            x = self.prelu(x)
            x = self.mask_conv1d(x)
        
        x = self.mask_nonlinear(x)
        output = x.view(batch_size, n_sources, out_channels, n_frames)
        
        return output

class Spectrogram(nn.Module):
    def __init__(self, fft_size, hop_size, n_mels, take_log=True):
        super().__init__()

        n_bins = fft_size // 2 + 1
        
        self.fft_size, self.hop_size = fft_size, hop_size
        self.n_mels = n_mels
        self.take_log = take_log

        self.window = nn.Parameter(torch.hann_window(fft_size), requires_grad=False)

        self.mel_transform = nn.Conv1d(n_bins, n_mels, kernel_size=1, stride=1, padding=0, bias=True)

        self.mean = nn.Parameter(torch.zeros(1, n_bins, 1), requires_grad=False)
        self.std = nn.Parameter(torch.ones(1, n_bins, 1), requires_grad=False)

        self.affine_bias = nn.Parameter(torch.zeros(1, n_bins, 1), requires_grad=True)
        self.affine_scale = nn.Parameter(torch.ones(1, n_bins, 1), requires_grad=True)

    def forward(self, input, length=None):
        magnitude = self.compute_magnitude(input)
        magnitude = (magnitude - self.mean) / self.std
        magnitude = self.affine_scale * magnitude + self.affine_bias
        output = self.mel_transform(magnitude)

        if length is not None:
            output = F.interpolate(output, size=length, mode='linear', align_corners=True)

        return output
    
    def compute_magnitude(self, wave):
        fft_size, hop_size = self.fft_size, self.hop_size

        spectrogram = torch.stft(wave, n_fft=fft_size, hop_length=hop_size, window=self.window, return_complex=True)
        magnitude = torch.abs(spectrogram)**2
    
        if self.take_log:
            magnitude = torch.log10(magnitude + 1e-12)

        return magnitude

class TemporalConvNet(nn.Module):
    def __init__(self, num_features, hidden_channels=256, skip_channels=256, kernel_size=3, num_blocks=3, num_layers=10, dilated=True, separable=False, causal=False, nonlinear=None, n_sources=2, conv_name='generated', norm_name='generated', eps=EPS, **kwargs):
        super().__init__()
        
        self.num_blocks = num_blocks
        
        net = []
        
        for idx in range(num_blocks):
            if idx == num_blocks - 1:
                net.append(ConvBlock1d(num_features, hidden_channels=hidden_channels, skip_channels=skip_channels, kernel_size=kernel_size, num_layers=num_layers, dilated=dilated, separable=separable, causal=causal, nonlinear=nonlinear, dual_head=False, n_sources=n_sources, conv_name=conv_name, norm_name=norm_name, eps=eps, **kwargs))
            else:
                net.append(ConvBlock1d(num_features, hidden_channels=hidden_channels, skip_channels=skip_channels, kernel_size=kernel_size, num_layers=num_layers, dilated=dilated, separable=separable, causal=causal, nonlinear=nonlinear, dual_head=True, n_sources=n_sources, conv_name=conv_name, norm_name=norm_name, eps=eps, **kwargs))
        
        self.net = nn.Sequential(*net)
    
    def forward(self, input, embedding=None):
        num_blocks = self.num_blocks
        
        x = input
        skip_connection = 0
        
        for idx in range(num_blocks):
            x, skip = self.net[idx](x, embedding=embedding)
            skip_connection = skip_connection + skip

        output = skip_connection
        
        return output

class ConvBlock1d(nn.Module):
    def __init__(
        self,
        num_features, hidden_channels=256, skip_channels=256,
        kernel_size=3, num_layers=10, dilated=True, separable=False, causal=False, nonlinear=None,
        dual_head=True, n_sources=2,
        conv_name='generated', norm_name='generated',
        eps=EPS,
        **kwargs
    ):
        super().__init__()
        
        self.num_layers = num_layers
        
        net = []
        
        for idx in range(num_layers):
            if dilated:
                dilation = 2**idx
                stride = 1
            else:
                dilation = 1
                stride = 2
            if not dual_head and idx == num_layers - 1:
                net.append(ResidualBlock1d(num_features, hidden_channels=hidden_channels, skip_channels=skip_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, separable=separable, causal=causal, nonlinear=nonlinear, dual_head=False, n_sources=n_sources, conv_name=conv_name, norm_name=norm_name, eps=eps, **kwargs))
            else:
                net.append(ResidualBlock1d(num_features, hidden_channels=hidden_channels, skip_channels=skip_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, separable=separable, causal=causal, nonlinear=nonlinear, dual_head=True, n_sources=n_sources, conv_name=conv_name, norm_name=norm_name, eps=eps, **kwargs))
            
        self.net = nn.Sequential(*net)

    def forward(self, input, embedding=None):
        num_layers = self.num_layers
        
        x = input
        skip_connection = 0
        
        for idx in range(num_layers):
            x, skip = self.net[idx](x, embedding=embedding)
            skip_connection = skip_connection + skip

        return x, skip_connection
        
class ResidualBlock1d(nn.Module):
    def __init__(self, num_features, hidden_channels=256, skip_channels=256, kernel_size=3, stride=2, dilation=1, separable=False, causal=False, nonlinear=None, dual_head=True, n_sources=2, conv_name='generated', norm_name='generated', eps=EPS, **kwargs):
        super().__init__()
        
        self.kernel_size, self.stride, self.dilation = kernel_size, stride, dilation
        self.separable, self.causal = separable, causal
        self.norm_name = norm_name
        self.dual_head = dual_head
        
        self.bottleneck_conv1d = choose_conv1d(conv_name, num_features, hidden_channels, kernel_size=1, stride=1, norm_name=norm_name, n_sources=n_sources, **kwargs)
        
        if nonlinear is not None:
            if nonlinear == 'prelu':
                self.nonlinear1d = nn.PReLU()
            else:
                raise ValueError("Not support {}".format(nonlinear))
            self.nonlinear = True
        else:
            self.nonlinear = False
        
        if norm_name:
            self.norm1d = choose_layer_norm(norm_name, hidden_channels, causal=causal, n_sources=n_sources, eps=eps, **kwargs)
        
        if separable:
            self.separable_conv1d = DepthwiseSeparableConv1d(
                hidden_channels, num_features, skip_channels=skip_channels,
                kernel_size=kernel_size, stride=stride, dilation=dilation,
                causal=causal, nonlinear=nonlinear,
                dual_head=dual_head,
                n_sources=n_sources, conv_name=conv_name, norm_name=norm_name, 
                eps=eps,
                **kwargs
            )
        else:
            if dual_head:
                self.output_conv1d = choose_conv1d(conv_name, hidden_channels, num_features, kernel_size=kernel_size, dilation=dilation, norm_name=norm_name, n_sources=n_sources, **kwargs)
            self.skip_conv1d = choose_conv1d(conv_name, hidden_channels, skip_channels, kernel_size=kernel_size, dilation=dilation, norm_name=norm_name, n_sources=n_sources, **kwargs)
        
    def forward(self, input, embedding=None):
        kernel_size, stride, dilation = self.kernel_size, self.stride, self.dilation
        nonlinear, norm_name = self.nonlinear, self.norm_name
        separable, causal = self.separable, self.causal
        dual_head = self.dual_head
        
        _, _, _, T_original = input.size()
        
        residual = input
        if embedding is not None:
            x = self.bottleneck_conv1d(input, embedding=embedding)
        else:
            x = self.bottleneck_conv1d(input)
        
        if nonlinear:
            x = self.nonlinear1d(x)
        if norm_name:
            if embedding is not None:
                x = self.norm1d(x, embedding=embedding)
            else:
                x = self.norm1d(x)
        
        padding = (T_original - 1) * stride - T_original + (kernel_size - 1) * dilation + 1
        
        if causal:
            padding_left = padding
            padding_right = 0
        else:
            padding_left = padding//2
            padding_right = padding - padding_left

        x = F.pad(x, (padding_left, padding_right))
        
        if separable:
            if embedding is not None:
                output, skip = self.separable_conv1d(x, embedding=embedding) # output may be None
            else:
                output, skip = self.separable_conv1d(x) # output may be None
        else:
            if dual_head:
                if embedding is not None:
                    output = self.output_conv1d(x, embedding=embedding)
                else:
                    output = self.output_conv1d(x)
            else:
                output = None
            
            if embedding is not None:
                skip = self.skip_conv1d(x, embedding=embedding)
            else:
                skip = self.skip_conv1d(x)
        
        if output is not None:
            output = output + residual
            
        return output, skip

class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels=256, skip_channels=256, kernel_size=3, stride=2, dilation=1, causal=False, nonlinear=None, dual_head=True, n_sources=2, conv_name='generated', norm_name='generated', eps=EPS, **kwargs):
        super().__init__()
        
        self.dual_head = dual_head
        self.norm_name = norm_name
        self.eps = eps
        
        self.depthwise_conv1d = choose_conv1d(conv_name, in_channels, in_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, groups=in_channels, n_sources=n_sources, **kwargs)
        
        if nonlinear is not None:
            if nonlinear == 'prelu':
                self.nonlinear1d = nn.PReLU()
            else:
                raise ValueError("Not support {}".format(nonlinear))
            self.nonlinear = True
        else:
            self.nonlinear = False
        
        if norm_name:
            self.norm1d = choose_layer_norm(norm_name, in_channels, causal=causal, n_sources=n_sources, eps=eps, **kwargs)

        if dual_head:
            self.output_pointwise_conv1d = choose_conv1d(conv_name, in_channels, out_channels, kernel_size=1, stride=1, n_sources=n_sources, **kwargs)
        
        self.skip_pointwise_conv1d = choose_conv1d(conv_name, in_channels, skip_channels, kernel_size=1, stride=1, n_sources=n_sources, **kwargs)
        
    def forward(self, input, embedding=None):
        """
        Args:
            input: (batch_size, C_in, T_in)
            embedding: (n_sources, embed_dim)
        Returns:
            output: (batch_size, C_out, T_out)
            skip: (batch_size, C_out, T_out) or None
        """
        nonlinear, norm_name = self.nonlinear, self.norm_name
        dual_head = self.dual_head
        
        if embedding is not None:
            x = self.depthwise_conv1d(input, embedding=embedding)
        else:
            x = self.depthwise_conv1d(input)
        
        if nonlinear:
            x = self.nonlinear1d(x)
        if norm_name:
            if embedding is not None:
                x = self.norm1d(x, embedding=embedding)
            else:
                x = self.norm1d(x)
        if dual_head:
            if embedding is not None:
                output = self.output_pointwise_conv1d(x, embedding=embedding)
            else:
                output = self.output_pointwise_conv1d(x)
        else:
            output = None
        
        if embedding is not None:
            skip = self.skip_pointwise_conv1d(x, embedding=embedding)
        else:
            skip = self.skip_pointwise_conv1d(x)
        
        return output, skip

class Conv1dGenerated(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, embed_dim=None, bottleneck_channels=None, n_sources=2):
        """
        Args:
            in_channels <int>: Input channels
            out_channels <int>: Output channels
            kernel_size <int>: Kernel size of 1D convolution
            stride <int>: Stride of 1D convolution
            padding <int>: Padding of 1D convolution
            dilation <int>: Dilation of 1D convolution
            groups <int>: Group of 1D convolution
            bias <bool>: Applies bias to 1D convolution
            embed_dim <int>: Embedding dimension
            bottleneck_channels <int>: Bottleneck channels
            n_sources <int>: Number of sources
        """
        super().__init__()

        self.in_channels, self.out_channels = in_channels, out_channels

        self.kernel_size, self.stride = kernel_size, stride
        self.padding, self.dilation = padding, dilation
        self.groups = groups
        self.bias = bias
        self.n_sources = n_sources
        
        self.bottleneck = nn.Linear(embed_dim, bottleneck_channels)
        self.linear = nn.Linear(bottleneck_channels, out_channels*in_channels//groups*kernel_size)
        self.linear_bias = nn.Linear(bottleneck_channels, out_channels)

    def forward(self, input, embedding):
        """
        Arguments:
            input <torch.Tensor>: (batch_size, n_sources, C_in, T_in)
            embedding <torch.Tensor>: (n_sources, embed_dim)
        Returns:
            output <torch.Tensor>: (batch_size, n_sources, C_out, T_out)
        """
        C_in, C_out = self.in_channels, self.out_channels
        kernel_size, stride = self.kernel_size, self.stride
        padding, dilation = self.padding, self.dilation
        groups = self.groups
        n_sources = self.n_sources

        batch_size, _, _, T_in = input.size()

        x_embedding = self.bottleneck(embedding)  # (n_sources, bottleneck_channels)
        kernel = self.linear(x_embedding)
        kernel = kernel.view(n_sources * C_out, C_in//groups, kernel_size)

        x = input.view(batch_size, n_sources * C_in, T_in) # (batch_size, n_sources * C_in, T_in)
        x = F.conv1d(x, kernel, bias=None, stride=stride, padding=padding, dilation=dilation, groups=n_sources*groups)  # (B, n_sources * C_out, T_out)
        x = x.view(batch_size, n_sources, C_out, -1)

        if self.bias:
            bias = self.linear_bias(x_embedding)
            bias = bias.view(1, n_sources, C_out, 1)
            output = x + bias  # (batch_size, n_sources, C_out, T_out)
        else:
            output = x

        return output

class Conv1dStatic(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, n_sources=2):
        """
        Args:
            in_channels <int>: Input channels
            out_channels <int>: Output channels
            kernel_size <int>: Kernel size of 1D convolution
            stride <int>: Stride of 1D convolution
            padding <int>: Padding of 1D convolution
            dilation <int>: Dilation of 1D convolution
            groups <int>: Group of 1D convolution
            bias <bool>: Applies bias to 1D convolution
            bottleneck_channels <int>: Bottleneck channels
            n_sources <int>: Number of sources
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_sources = n_sources
        
        self.conv1d = nn.Conv1d(n_sources*in_channels, n_sources*out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=n_sources*groups, bias=bias)

    def forward(self, input):
        """
        Args:
            input (batch_size, n_sources, C_in, T_in)
        Returns:
            output (batch_size, n_sources, C_out, T_out)
        """
        C_in, C_out = self.in_channels, self.out_channels
        n_sources = self.n_sources

        batch_size, _, _, T_in = input.size()

        x = input.view(batch_size, n_sources*C_in, T_in)  # (batch_size, n_sources*C_in, T_in)
        x = self.conv1d(x)  # (batch_size, n_sources*C_out, T_out)
        output = x.view(batch_size, n_sources, C_out, -1)  # (batch_size, n_sources, C_out, T_out)

        return output

class GroupNormGenerated(nn.Module):
    def __init__(self, num_features, groups=1, embed_dim=None, bottleneck_channels=None, n_sources=2, eps=EPS):
        super().__init__()

        self.groups = groups
        self.num_features = num_features
        self.n_sources = n_sources
        self.eps = eps

        self.bottleneck = nn.Linear(embed_dim, bottleneck_channels)
        self.linear_scale = nn.Linear(bottleneck_channels, num_features)
        self.linear_bias = nn.Linear(bottleneck_channels, num_features)

    def forward(self, input, embedding):
        """
        Args:
            input: (batch_size, n_sources, C, T)
            embed_dim: (n_sources, embed_dim)
        Returns:
            output (batch_size, n_sources, C, T)
        """
        batch_size, _, _, T = input.size()
        num_features, groups = self.num_features, self.groups
        n_sources = self.n_sources

        x_embedding = self.bottleneck(embedding)  # (n_sources, bottleneck_channels)
        scale = self.linear_scale(x_embedding) # (n_sources, C)
        bias = self.linear_bias(x_embedding) # (n_sources, C)

        scale, bias = scale.view(-1), bias.view(-1) # (n_sources * C,), (n_sources * C,)

        x = input.view(batch_size, n_sources * num_features, T) # (batch_size, n_sources * C, T)
        x = F.group_norm(x, n_sources * groups, weight=scale, bias=bias, eps=self.eps)  # (batch_size, n_sources * C, T)
        output = x.view(batch_size, n_sources, num_features, T) # (batch_size, n_sources, C, T)

        return output

class GroupNormStatic(nn.Module):
    def __init__(self, num_features, groups=1, n_sources=2, eps=EPS):
        super().__init__()

        self.num_features = num_features
        self.n_sources = n_sources

        self.norm = nn.GroupNorm(n_sources*groups, n_sources*num_features, eps=eps)

    def forward(self, input):
        """
        Args:
            input: (batch_size, n_sources, C, T)
        Returns:
            output (batch_size, n_sources, C, T)
        """
        n_sources = self.n_sources
        num_features = self.num_features
        batch_size, _, _, T = input.size()

        x = input.view(batch_size, n_sources*num_features, T)
        x = self.norm(x)
        output = x.view(batch_size, n_sources, num_features, T)

        return output

def choose_conv1d(name, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, n_sources=2, **kwargs):
    if name == 'generated':
        embed_dim, bottleneck_channels = kwargs['embed_dim'], kwargs['bottleneck_channels']
        conv1d = Conv1dGenerated(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, embed_dim=embed_dim, bottleneck_channels=bottleneck_channels, n_sources=n_sources)
    elif name == 'static':
        conv1d = Conv1dStatic(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, n_sources=n_sources)
    else:
        raise NotImplementedError("Not support {} convolution.".format(name))
    return conv1d

def choose_layer_norm(name, num_features, causal=False, eps=EPS, **kwargs):
    assert not causal, "Causal should be False"

    groups = kwargs.get('groups') or 1
    n_sources = kwargs['n_sources']

    if name == 'generated':
        embed_dim, bottleneck_channels = kwargs['embed_dim'], kwargs['bottleneck_channels']
        layer_norm = GroupNormGenerated(num_features, groups=groups, embed_dim=embed_dim, bottleneck_channels=bottleneck_channels, n_sources=n_sources, eps=eps)
    elif name == 'static':
        layer_norm = GroupNormStatic(num_features, groups=groups, n_sources=n_sources, eps=eps)
    else:
        raise NotImplementedError("Not support {} layer normalization.".format(name))
    
    return layer_norm

def get_kwargs_meta(kwargs):
    kwargs_meta = {}

    for key in kwargs.keys():
        if key[:6] == 'embed_':
            if key == 'embed_dim':
                kwargs_meta[key] = kwargs[key]
            else:
                """
                Example:
                    If `kwargs` has ['embed_groups'], then kwargs_meta['group'] = kwargs['groups']
                """
                key_tcn = key.replace('embed_', '')
                kwargs_meta[key_tcn] = kwargs[key]
    
    return kwargs_meta

def _test_conv1d():
    batch_size, n_sources = 2, 4
    C_in, T_in = 3, 10
    C_out = 5
    embed_dim = 8

    input, embedding = torch.randn(batch_size, n_sources, C_in, T_in), torch.randn(n_sources, embed_dim)
    conv1d = Conv1dGenerated(C_in, C_out, kernel_size=3, embed_dim=embed_dim, bottleneck_channels=6, n_sources=n_sources)
    output = conv1d(input, embedding)

    print(conv1d)
    print(input.size(), embedding.size(), output.size())

    input = torch.randn(batch_size, n_sources, C_in, T_in)
    conv1d = Conv1dStatic(C_in, C_out, kernel_size=3, n_sources=n_sources)
    output = conv1d(input)

    print(conv1d)
    print(input.size(), output.size())

def _test_tcn():
    batch_size = 4
    T = 128
    in_channels, out_channels, skip_channels = 16, 16, 32
    kernel_size, stride = 3, 1
    num_blocks = 3
    num_layers = 4
    dilated, separable = True, False
    causal = False
    nonlinear = 'prelu'
    conv_name, norm_name = 'generated', 'generated'
    n_sources = 3
    embed_dim, bottleneck_channels = 8, 10

    print("-"*10, "Meta-TasNet (Generated)", "-"*10)
    input, embedding = torch.randn((batch_size, n_sources, in_channels, T), dtype=torch.float), torch.randn((n_sources, embed_dim), dtype=torch.float)
    
    model = TemporalConvNet(
        in_channels, hidden_channels=out_channels, skip_channels=skip_channels,
        kernel_size=kernel_size,
        num_blocks=num_blocks, num_layers=num_layers,
        dilated=dilated, separable=separable, causal=causal, nonlinear=nonlinear,
        conv_name=conv_name, norm_name=norm_name,
        n_sources=n_sources, embed_dim=embed_dim, bottleneck_channels=bottleneck_channels
    )
    
    print(model)
    output = model(input, embedding=embedding)
    print(input.size(), embedding.size(), output.size())

    print("-"*10, "Meta-TasNet (Generated)", "-"*10)
    conv_name, norm_name = 'static', 'static'

    input = torch.randn((batch_size, n_sources, in_channels, T), dtype=torch.float)
    
    model = TemporalConvNet(
        in_channels, hidden_channels=out_channels, skip_channels=skip_channels,
        kernel_size=kernel_size,
        num_blocks=num_blocks, num_layers=num_layers,
        dilated=dilated, separable=separable, causal=causal, nonlinear=nonlinear,
        conv_name=conv_name, norm_name=norm_name,
        n_sources=n_sources
    )
    
    print(model)
    output = model(input)
    print(input.size(), output.size())

def _test_separator():
    batch_size = 4
    T = 128
    n_sources = 3
    
    B, H, Sc = 8, 10, 12
    P = 3
    R, X = 2, 4

    print("-"*10, "Separator (Generated)", "-"*10)

    D_l, B_l = 6, 5
    embedding = torch.randn((n_sources, D_l), dtype=torch.float)

    N_base = 6

    N = N_base
    N_in, N_out = N, N
    input0 = torch.randn((batch_size, n_sources, N, T), dtype=torch.float)
    
    model = Separator(
        N_in, N_out, bottleneck_channels=B, hidden_channels=H, skip_channels=Sc,
        kernel_size=P, num_blocks=R, num_layers=X,
        causal=False, mask_nonlinear='softmax',
        conv_name='generated', norm_name='generated',
        n_sources=n_sources,
        embed_dim=D_l, embed_bottleneck_channels=B_l
    )

    print(model)
    output0 = model(input0, embedding=embedding)
    print(input0.size(), embedding.size(), output0.size())

    N = 2 * N_base
    N_in, N_out = N + N // 2, N
    input1 = torch.randn((batch_size, n_sources, N, T), dtype=torch.float)
    
    model = Separator(
        N_in, N_out, bottleneck_channels=B, hidden_channels=H, skip_channels=Sc,
        kernel_size=P, num_blocks=R, num_layers=X,
        causal=False, mask_nonlinear='softmax',
        conv_name='generated', norm_name='generated',
        n_sources=n_sources,
        embed_dim=D_l, embed_bottleneck_channels=B_l
    )

    print(model)
    input1 = torch.cat([output0, input1], dim=2)
    output = model(input1, embedding=embedding)
    print(input1.size(), embedding.size(), output.size())

def _test_meta_tasnet_backbone():
    T = 2**15
    n_sources = 4
    D_l, B_l = 6, 5
    
    B, H, Sc = 8, 10, 12
    P = 3
    R, X = 2, 4

    wave1, sample_rate = torchaudio.load("../../dataset/sample-song/single-channel/sample-2_piano_16000.wav")
    wave2, sample_rate = torchaudio.load("../../dataset/sample-song/single-channel/sample-2_violin_16000.wav")
    input = torch.cat([wave1.unsqueeze(dim=0), wave2.unsqueeze(dim=0)], dim=0)
    input = input[:, :, :T]

    print(input.size())

    K, S = 20, 6
    F, M = 3, 256
    N = 32

    print("-"*10, "Meta-TasNet Backbone (Generated, sample_rate = 8000)", "-"*10)

    sample_rate0 = 8000
    resampler0 = torchaudio.transforms.Resample(sample_rate, sample_rate0)

    K0, S0 = K, S
    N0 = N
    F0 = F
    enc_L0, enc_S0 = 1024 * (sample_rate0//8000), 256 * (sample_rate0//8000)

    input0 = resampler0(input)

    model = MetaTasNetBackbone(
        N0, K0, stride=S0,
        enc_fft_size=enc_L0, enc_hop_size=enc_S0, num_filters=F0, n_mels=M,
        sep_hidden_channels=H, sep_bottleneck_channels=B, sep_skip_channels=Sc,
        sep_kernel_size=P, sep_num_blocks=X, sep_num_layers=R,
        conv_name='generated', norm_name='generated',
        n_sources=n_sources,
        embed_dim=D_l, embed_bottleneck_channels=B_l
    )
    
    print(model)
    output0, latent0 = model.extract_latent(input0)
    print(input0.size(), latent0.size(), output0.size())
    print()

    print("-"*10, "Meta-TasNet Backbone (Generated, sample_rate = 8000 + 16000)", "-"*10)

    sample_rate1 = 16000
    resampler1 = torchaudio.transforms.Resample(sample_rate, sample_rate1)

    K1, S1 = 2 * K, 2 * S
    N1 = 2 * N
    F1 = 2 * F
    enc_L1, enc_S1 = 1024 * (sample_rate1//8000), 256 * (sample_rate1//8000)

    input1 = resampler1(input)

    model = MetaTasNetBackbone(
        N1, K1, stride=S1,
        enc_fft_size=enc_L1, enc_hop_size=enc_S1, num_filters=F1, n_mels=M,
        sep_in_channels=N0+N1, sep_hidden_channels=H, sep_bottleneck_channels=B, sep_skip_channels=Sc,
        sep_kernel_size=P, sep_num_blocks=X, sep_num_layers=R,
        conv_name='generated', norm_name='generated',
        n_sources=n_sources,
        embed_dim=D_l, embed_bottleneck_channels=B_l
    )
    
    print(model)
    output1, latent1 = model.extract_latent(input1, latent=latent0)
    print(input1.size(), latent1.size(), output1.size())
    print()

def _test_meta_tasnet():
    T = 2**15
    n_sources = 4
    D_l, B_l = 6, 5
    
    B, H, Sc = 8, 10, 12
    P = 3
    R, X = 2, 4

    wave1, sample_rate = torchaudio.load("../../dataset/sample-song/single-channel/sample-2_piano_16000.wav")
    wave2, sample_rate = torchaudio.load("../../dataset/sample-song/single-channel/sample-2_violin_16000.wav")
    input_original = torch.cat([wave1.unsqueeze(dim=0), wave2.unsqueeze(dim=0)], dim=0)
    input_original = input_original[:, :, :T]

    print(input_original.size())

    print("-"*10, "Meta-TasNet (Generated, base sample_rate = 8000)", "-"*10)

    sample_rate_original = 16000
    sample_rate = [8000, 16000, 32000]
    input = []

    for sample_rate_target in sample_rate:
        resampler = torchaudio.transforms.Resample(sample_rate_original, sample_rate_target)
        _input = resampler(input_original)
        input.append(_input)

    num_stages = len(sample_rate)
    K, S = 20, 6
    F, M = 3, 256
    N = 32
    fft_size, hop_size = 1024 * (sample_rate[0]//8000), 256 * (sample_rate[0]//8000)

    model = MetaTasNet(
        N, K, stride=S,
        enc_fft_size=fft_size, enc_hop_size=hop_size, num_filters=F, n_mels=M,
        sep_hidden_channels=H, sep_bottleneck_channels=B, sep_skip_channels=Sc, sep_kernel_size=P, sep_num_blocks=X, sep_num_layers=R,
        conv_name='generated', norm_name='generated',
        num_stages=num_stages, n_sources=n_sources,
        embed_dim=D_l, embed_bottleneck_channels=B_l
    )
    
    print(model)
    print(model.num_parameters)
    output = model(input)

    for _input, _output in zip(input, output):
        print(_input.size(), _output.size())

    print("-"*10, "No masking", "-"*10)

    model = MetaTasNet(
        N, K, stride=S,
        enc_fft_size=fft_size, enc_hop_size=hop_size, num_filters=F, n_mels=M,
        sep_hidden_channels=H, sep_bottleneck_channels=B, sep_skip_channels=Sc, sep_kernel_size=P, sep_num_blocks=X, sep_num_layers=R,
        conv_name='generated', norm_name='generated',
        num_stages=num_stages, n_sources=n_sources,
        embed_dim=D_l, embed_bottleneck_channels=B_l
    )

    print(model)
    print(model.num_parameters)
    output = model(input, masking=False)

    for _input, _output in zip(input, output):
        print(_input.size(), _output.size())

if __name__ == '__main__':
    import torchaudio

    torch.manual_seed(111)

    print('='*10, "Conv1d", '='*10)
    _test_conv1d()
    print()

    print('='*10, "TCN", '='*10)
    _test_tcn()
    print()

    print('='*10, "Separator", '='*10)
    _test_separator()
    print()

    print('='*10, "MetaTasNet backbone", '='*10)
    _test_meta_tasnet_backbone()
    print()

    print('='*10, "MetaTasNet", '='*10)
    _test_meta_tasnet()