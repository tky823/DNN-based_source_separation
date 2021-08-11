import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-12

class MetaTasNet(nn.Module):
    def __init__(self):
        pass

class Separator(nn.Module):
    def __init__(
        self,
        in_channels, out_channels, bottleneck_channels=128, hidden_channels=256, skip_channels=128,
        kernel_size=3, num_blocks=3, num_layers=8, dilated=True, separable=True, causal=True, nonlinear='prelu', mask_nonlinear='softmax',
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
        
    def forward(self, input, latent=None, input_partial=None):
        """
        Args:
            input (batch_size, n_sources, in_channels, n_frames)
        Returns:
            output (batch_size, n_sources, out_channels, n_frames)
        """
        out_channels, n_sources = self.out_channels, self.n_sources

        batch_size, _, _, n_frames = input.size()

        if input_partial is not None:
            x = torch.cat([input, input_partial], dim=2)
        else:
            x = input
        
        if latent is not None:
            x = self.norm1d(x, latent=latent)
            x = self.bottleneck_conv1d(x, latent=latent)
            x = self.tcn(x, latent=latent)
            x = self.prelu(x)
            x = self.mask_conv1d(x, latent=latent)
        else:
            x = self.norm1d(x)
            x = self.bottleneck_conv1d(x)
            x = self.tcn(x)
            x = self.prelu(x)
            x = self.mask_conv1d(x)
        
        x = self.mask_nonlinear(x)
        output = x.view(batch_size, n_sources, out_channels, n_frames)
        
        return output

class TemporalConvNet(nn.Module):
    def __init__(self, num_features, hidden_channels=256, skip_channels=256, kernel_size=3, num_blocks=3, num_layers=10, dilated=True, separable=False, causal=True, nonlinear=None, n_sources=2, conv_name='generated', norm_name='generated', eps=EPS, **kwargs):
        super().__init__()
        
        self.num_blocks = num_blocks
        
        net = []
        
        for idx in range(num_blocks):
            if idx == num_blocks - 1:
                net.append(ConvBlock1d(num_features, hidden_channels=hidden_channels, skip_channels=skip_channels, kernel_size=kernel_size, num_layers=num_layers, dilated=dilated, separable=separable, causal=causal, nonlinear=nonlinear, dual_head=False, n_sources=n_sources, conv_name=conv_name, norm_name=norm_name, eps=eps, **kwargs))
            else:
                net.append(ConvBlock1d(num_features, hidden_channels=hidden_channels, skip_channels=skip_channels, kernel_size=kernel_size, num_layers=num_layers, dilated=dilated, separable=separable, causal=causal, nonlinear=nonlinear, dual_head=True, n_sources=n_sources, conv_name=conv_name, norm_name=norm_name, eps=eps, **kwargs))
        
        self.net = nn.Sequential(*net)
    
    def forward(self, input, latent=None):
        num_blocks = self.num_blocks
        
        x = input
        skip_connection = 0
        
        for idx in range(num_blocks):
            x, skip = self.net[idx](x, latent=latent)
            skip_connection = skip_connection + skip

        output = skip_connection
        
        return output

class ConvBlock1d(nn.Module):
    def __init__(
        self,
        num_features, hidden_channels=256, skip_channels=256,
        kernel_size=3, num_layers=10, dilated=True, separable=False, causal=True, nonlinear=None,
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

    def forward(self, input, latent=None):
        num_layers = self.num_layers
        
        x = input
        skip_connection = 0
        
        for idx in range(num_layers):
            x, skip = self.net[idx](x, latent=latent)
            skip_connection = skip_connection + skip

        return x, skip_connection
        
class ResidualBlock1d(nn.Module):
    def __init__(self, num_features, hidden_channels=256, skip_channels=256, kernel_size=3, stride=2, dilation=1, separable=False, causal=True, nonlinear=None, dual_head=True, n_sources=2, conv_name='generated', norm_name='generated', eps=EPS, **kwargs):
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
        
    def forward(self, input, latent=None):
        kernel_size, stride, dilation = self.kernel_size, self.stride, self.dilation
        nonlinear, norm_name = self.nonlinear, self.norm_name
        separable, causal = self.separable, self.causal
        dual_head = self.dual_head
        
        _, _, _, T_original = input.size()
        
        residual = input
        if latent is not None:
            x = self.bottleneck_conv1d(input, latent=latent)
        else:
            x = self.bottleneck_conv1d(input)
        
        if nonlinear:
            x = self.nonlinear1d(x)
        if norm_name:
            if latent is not None:
                x = self.norm1d(x, latent=latent)
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
            if latent is not None:
                output, skip = self.separable_conv1d(x, latent=latent) # output may be None
            else:
                output, skip = self.separable_conv1d(x) # output may be None
        else:
            if dual_head:
                if latent is not None:
                    output = self.output_conv1d(x, latent=latent)
                else:
                    output = self.output_conv1d(x)
            else:
                output = None
            
            if latent is not None:
                skip = self.skip_conv1d(x, latent=latent)
            else:
                skip = self.skip_conv1d(x)
        
        if output is not None:
            output = output + residual
            
        return output, skip

class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels=256, skip_channels=256, kernel_size=3, stride=2, dilation=1, causal=True, nonlinear=None, dual_head=True, n_sources=2, conv_name='generated', norm_name='generated', eps=EPS, **kwargs):
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
        
    def forward(self, input, latent=None):
        """
        Args:
            input: (batch_size, C_in, T_in)
            latent: (n_sources, latent_dim)
        Returns:
            output: (batch_size, C_out, T_out)
            skip: (batch_size, C_out, T_out) or None
        """
        nonlinear, norm_name = self.nonlinear, self.norm_name
        dual_head = self.dual_head
        
        if latent is not None:
            x = self.depthwise_conv1d(input, latent=latent)
        else:
            x = self.depthwise_conv1d(input)
        
        if nonlinear:
            x = self.nonlinear1d(x)
        if norm_name:
            if latent is not None:
                x = self.norm1d(x, latent=latent)
            else:
                x = self.norm1d(x)
        if dual_head:
            if latent is not None:
                output = self.output_pointwise_conv1d(x, latent=latent)
            else:
                output = self.output_pointwise_conv1d(x)
        else:
            output = None
        
        if latent is not None:
            skip = self.skip_pointwise_conv1d(x, latent=latent)
        else:
            skip = self.skip_pointwise_conv1d(x)
        
        return output, skip

class Conv1dGenerated(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, latent_dim=None, bottleneck_channels=None, n_sources=2):
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
            latent_dim <int>: Embedding dimension
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
        
        self.bottleneck = nn.Linear(latent_dim, bottleneck_channels)
        self.linear = nn.Linear(bottleneck_channels, out_channels*in_channels//groups*kernel_size)
        self.linear_bias = nn.Linear(bottleneck_channels, out_channels)

    def forward(self, input, latent):
        """
        Arguments:
            input <torch.Tensor>: (batch_size, n_sources, C_in, T_in)
            latent <torch.Tensor>: (n_sources, latent_dim)
        Returns:
            output <torch.Tensor>: (batch_size, n_sources, C_out, T_out)
        """
        C_in, C_out = self.in_channels, self.out_channels
        kernel_size, stride = self.kernel_size, self.stride
        padding, dilation = self.padding, self.dilation
        groups = self.groups
        n_sources = self.n_sources

        batch_size, _, _, T_in = input.size()

        x_latent = self.bottleneck(latent)  # (n_sources, bottleneck_channels)
        kernel = self.linear(x_latent)
        kernel = kernel.view(n_sources * C_out, C_in//groups, kernel_size)

        x = input.view(batch_size, n_sources * C_in, T_in)  # shape: (batch_size, n_sources * C_in, T_in)
        x = F.conv1d(x, kernel, bias=None, stride=stride, padding=padding, dilation=dilation, groups=n_sources*groups)  # shape: (B, n_sources * C_out, T_out)
        x = x.view(batch_size, n_sources, C_out, -1)

        if self.bias:
            bias = self.linear_bias(x_latent)
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
    def __init__(self, num_features, groups=1, latent_dim=None, bottleneck_channels=None, n_sources=2, eps=EPS):
        super().__init__()

        self.groups = groups
        self.num_features = num_features
        self.n_sources = n_sources
        self.eps = eps

        self.bottleneck = nn.Linear(latent_dim, bottleneck_channels)
        self.linear_scale = nn.Linear(bottleneck_channels, num_features)
        self.linear_bias = nn.Linear(bottleneck_channels, num_features)

    def forward(self, input, latent):
        """
        Args:
            input: (batch_size, n_sources, C, T)
            latent: (n_sources, latent)
        Returns:
            output (batch_size, n_sources, C, T)
        """
        batch_size, _, _, T = input.size()
        num_features, groups = self.num_features, self.groups
        n_sources = self.n_sources

        x_latent = self.bottleneck(latent)  # (n_sources, bottleneck_channels)
        scale = self.linear_scale(x_latent) # (n_sources, C)
        bias = self.linear_bias(x_latent) # (n_sources, C)

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
        latent_dim, bottleneck_channels = kwargs['latent_dim'], kwargs['bottleneck_channels']
        conv1d = Conv1dGenerated(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, latent_dim=latent_dim, bottleneck_channels=bottleneck_channels, n_sources=n_sources)
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
        latent_dim, bottleneck_channels = kwargs['latent_dim'], kwargs['bottleneck_channels']
        layer_norm = GroupNormGenerated(num_features, groups=groups, latent_dim=latent_dim, bottleneck_channels=bottleneck_channels, n_sources=n_sources, eps=eps)
    elif name == 'static':
        layer_norm = GroupNormStatic(num_features, groups=groups, n_sources=n_sources, eps=eps)
    else:
        raise NotImplementedError("Not support {} layer normalization.".format(name))
    
    return layer_norm

def get_kwargs_meta(kwargs):
    kwargs_meta = {}

    for key in kwargs.keys():
        if key[:7] == 'latent_':
            if key == 'latent_dim':
                kwargs_meta[key] = kwargs[key]
            else:
                """
                Example:
                    If `kwargs` has ['latent_groups'], then kwargs_meta['group'] = kwargs['groups']
                """
                key_tcn = key.replace('latent_', '')
                kwargs_meta[key_tcn] = kwargs[key]
    
    return kwargs_meta

def _test_conv1d():
    batch_size, n_sources = 2, 4
    C_in, T_in = 3, 10
    C_out = 5
    latent_dim = 8

    input, latent = torch.randn(batch_size, n_sources, C_in, T_in), torch.randn(n_sources, latent_dim)
    conv1d = Conv1dGenerated(C_in, C_out, kernel_size=3, latent_dim=latent_dim, bottleneck_channels=6, n_sources=n_sources)
    output = conv1d(input, latent)

    print(conv1d)
    print(input.size(), latent.size(), output.size())

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
    latent_dim, bottleneck_channels = 8, 10

    print("-"*10, "Meta-TasNet (Generated)", "-"*10)
    input, latent = torch.randn((batch_size, n_sources, in_channels, T), dtype=torch.float), torch.randn((n_sources, latent_dim), dtype=torch.float)
    
    model = TemporalConvNet(
        in_channels, hidden_channels=out_channels, skip_channels=skip_channels,
        kernel_size=kernel_size,
        num_blocks=num_blocks, num_layers=num_layers,
        dilated=dilated, separable=separable, causal=causal, nonlinear=nonlinear,
        conv_name=conv_name, norm_name=norm_name,
        n_sources=n_sources, latent_dim=latent_dim, bottleneck_channels=bottleneck_channels
    )
    
    print(model)
    output = model(input, latent=latent)
    print(input.size(), latent.size(), output.size())

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
    
    print("-"*10, "Meta-TasNet (Generated)", "-"*10)

    N_in, N_out = 6, 6
    D_l, B_l = 6, 5
    input, latent = torch.randn((batch_size, n_sources, N_in, T), dtype=torch.float), torch.randn((n_sources, D_l), dtype=torch.float)
    
    model = Separator(
        N_in, N_out, bottleneck_channels=B, hidden_channels=H, skip_channels=Sc,
        kernel_size=P, num_blocks=R, num_layers=X,
        causal=False, mask_nonlinear='softmax',
        conv_name='generated', norm_name='generated',
        n_sources=n_sources,
        latent_dim=D_l, latent_bottleneck_channels=B_l
    )

    print(model)
    output = model(input, latent=latent)
    print(input.size(), latent.size(), output.size())
    print()

    print("-"*10, "Meta-TasNet (Generated)", "-"*10)

    N = 6
    N_in, N_out = N + N // 2, N // 2
    D_l, B_l = 6, 5
    input, latent, input_partial = torch.randn((batch_size, n_sources, N, T), dtype=torch.float), torch.randn((n_sources, D_l), dtype=torch.float), torch.randn((batch_size, n_sources, N // 2, T), dtype=torch.float)
    
    model = Separator(
        N_in, N_out, bottleneck_channels=B, hidden_channels=H, skip_channels=Sc,
        kernel_size=P, num_blocks=R, num_layers=X,
        causal=False, mask_nonlinear='softmax',
        conv_name='generated', norm_name='generated',
        n_sources=n_sources,
        latent_dim=D_l, latent_bottleneck_channels=B_l
    )

    print(model)
    output = model(input, latent=latent, input_partial=input_partial)
    print(input.size(), latent.size(), input_partial.size(), output.size())

if __name__ == '__main__':
    torch.manual_seed(111)

    print('='*10, "Conv1d", '='*10)
    _test_conv1d()
    print()

    print('='*10, "TCN", '='*10)
    _test_tcn()
    print()

    print('='*10, "Separator", '='*10)
    _test_separator()