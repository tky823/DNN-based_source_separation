import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils_tasnet import choose_bases, choose_layer_norm
from models.gtu import GTU1d
from models.transform import Segment1d, OverlapAdd1d
from models.galr import GALR

EPS=1e-12

class GALRNet(nn.Module):
    def __init__(self, n_bases, kernel_size, stride=None, enc_bases=None, dec_bases=None, sep_hidden_channels=128, sep_bottleneck_channels=64, sep_chunk_size=100, sep_hop_size=50, sep_num_blocks=6, sep_num_heads=4, causal=True, sep_norm=True, mask_nonlinear='sigmoid', n_sources=2, eps=EPS, **kwargs):
        super().__init__()
        
        if stride is None:
            stride = kernel_size//2
        
        assert kernel_size%stride == 0, "kernel_size is expected divisible by stride"
        
        # Encoder-decoder
        self.n_bases = n_bases
        self.kernel_size, self.stride = kernel_size, stride
        self.enc_bases, self.dec_bases = enc_bases, dec_bases
        
        if enc_bases == 'trainable' and not dec_bases == 'pinv':    
            self.enc_nonlinear = kwargs['enc_nonlinear']
        else:
            self.enc_nonlinear = None
        
        if enc_bases in ['Fourier', 'trainableFourier'] or dec_bases in ['Fourier', 'trainableFourier']:
            self.window_fn = kwargs['window_fn']
        else:
            self.window_fn = None
        
        # Separator configuration
        self.sep_hidden_channels, self.sep_bottleneck_channels = sep_hidden_channels, sep_bottleneck_channels
        self.sep_chunk_size, self.sep_hop_size = sep_chunk_size, sep_hop_size
        self.sep_num_blocks = sep_num_blocks
        
        self.causal = causal
        self.sep_norm = sep_norm
        self.mask_nonlinear = mask_nonlinear
        
        self.n_sources = n_sources
        self.eps = eps
        
        # Network configuration
        encoder, decoder = choose_bases(n_bases, kernel_size=kernel_size, stride=stride, enc_bases=enc_bases, dec_bases=dec_bases, **kwargs)
        
        self.encoder = encoder
        self.separator = Separator(n_bases, hidden_channels=sep_hidden_channels, chunk_size=sep_chunk_size, hop_size=sep_hop_size, num_blocks=sep_num_blocks, num_heads=sep_num_heads, causal=causal, norm=sep_norm, mask_nonlinear=mask_nonlinear, n_sources=n_sources, eps=eps)
        self.decoder = decoder
        
        self.num_parameters = self._get_num_parameters()
        
    def forward(self, input):
        output, latent = self.extract_latent(input)
        
        return output
        
    def extract_latent(self, input):
        """
        Args:
            input (batch_size, 1, T)
        Returns:
            output (batch_size, n_sources, T)
            latent (batch_size, n_sources, n_bases, T'), where T' = (T-K)//S+1
        """
        n_sources = self.n_sources
        n_bases = self.n_bases
        kernel_size, stride = self.kernel_size, self.stride
        
        batch_size, C_in, T = input.size()
        
        assert C_in == 1, "input.size() is expected (?,1,?), but given {}".format(input.size())
        
        padding = (stride - (T-kernel_size)%stride)%stride
        padding_left = padding//2
        padding_right = padding - padding_left

        input = F.pad(input, (padding_left, padding_right))
        w = self.encoder(input)
        mask = self.separator(w)
        w = w.unsqueeze(dim=1)
        w_hat = w * mask
        latent = w_hat
        w_hat = w_hat.view(batch_size*n_sources, n_bases, -1)
        x_hat = self.decoder(w_hat)
        x_hat = x_hat.view(batch_size, n_sources, -1)
        output = F.pad(x_hat, (-padding_left, -padding_right))
        
        return output, latent
    
    def _get_num_parameters(self):
        num_parameters = 0
        
        for p in self.parameters():
            if p.requires_grad:
                num_parameters += p.numel()
                
        return num_parameters

class Separator(nn.Module):
    def __init__(self, num_features, bottleneck_channels=64, hidden_channels=128, chunk_size=100, hop_size=50, num_blocks=6, num_heads=4, causal=True, norm=True, mask_nonlinear='sigmoid', n_sources=2, eps=EPS):
        super().__init__()
        
        self.num_features, self.n_sources = num_features, n_sources
        self.chunk_size, self.hop_size = chunk_size, hop_size
        
        self.bottleneck_conv1d = nn.Conv1d(num_features, bottleneck_channels, kernel_size=1, stride=1)
        self.segment1d = Segment1d(chunk_size, hop_size)
        self.norm2d = choose_layer_norm(bottleneck_channels, causal=causal, eps=eps)
        self.galr = GALR(bottleneck_channels, hidden_channels, num_blocks=num_blocks, num_heads=num_heads, causal=causal, norm=norm, eps=eps)
        self.overlap_add1d = OverlapAdd1d(chunk_size, hop_size)
        self.gtu = GTU1d(bottleneck_channels, n_sources*num_features)
        
        if mask_nonlinear == 'relu':
            self.mask_nonlinear = nn.ReLU()
        elif mask_nonlinear == 'sigmoid':
            self.mask_nonlinear = nn.Sigmoid()
        elif mask_nonlinear == 'softmax':
            self.mask_nonlinear = nn.Softmax(dim=1)
        else:
            raise ValueError("Cannot support {}".format(mask_nonlinear))
            
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
        
        padding = (hop_size-(n_frames-chunk_size)%hop_size)%hop_size
        padding_left = padding//2
        padding_right = padding - padding_left
        
        x = self.bottleneck_conv1d(input)
        x = F.pad(x, (padding_left, padding_right))
        x = self.segment1d(x) # -> (batch_size, C, S, chunk_size)
        x = self.norm2d(x)
        x = self.galr(x)
        x = self.overlap_add1d(x)
        x = F.pad(x, (-padding_left, -padding_right))
        x = self.gtu(x)
        x = self.mask_nonlinear(x)
        output = x.view(batch_size, n_sources, num_features, n_frames)
        
        return output

def _test_separator():
    batch_size, n_frames = 2, 5
    N, H = 16, 32 # H is the number of channels for each direction
    K, P = 3, 2
    B = 3
    
    sep_norm = True
    mask_nonlinear = 'sigmoid'
    
    causal = True
    n_sources = 2
    
    input = torch.randn((batch_size, N, n_frames), dtype=torch.float)
    
    separator = Separator(N, hidden_channels=H, chunk_size=K, hop_size=P, num_blocks=B, causal=causal, norm=sep_norm, mask_nonlinear=mask_nonlinear, n_sources=n_sources)
    print(separator)

    output = separator(input)
    print(input.size(), output.size())

def _test_galrnet():
    batch_size, n_frames = 2, 5
    K, P = 3, 2

    # Encoder & decoder
    C, T = 1, 128
    L, N = 8, 16
    
    # Separator
    f = N
    H = 32 # for each direction
    B = 4
    sep_norm = True
    
    input = torch.randn((batch_size, C, T), dtype=torch.float)
    
    print("-"*10, "Trainable Bases & Non causal", "-"*10)
    enc_bases, dec_bases = 'trainable', 'trainable'
    enc_nonlinear = 'relu'
    
    causal = False
    mask_nonlinear = 'sigmoid'
    n_sources = 2
    
    model = GALRNet(N, kernel_size=L, enc_bases=enc_bases, dec_bases=dec_bases, enc_nonlinear=enc_nonlinear, sep_hidden_channels=H, sep_bottleneck_channels=f, sep_chunk_size=K, sep_hop_size=P, sep_num_blocks=B, causal=causal, sep_norm=sep_norm, mask_nonlinear=mask_nonlinear, n_sources=n_sources)
    print(model)
    print("# Parameters: {}".format(model.num_parameters))
    
    output = model(input)
    print(input.size(), output.size())
    print()
    
    print("-"*10, "Fourier Bases & Causal", "-"*10)
    enc_bases, dec_bases = 'Fourier', 'Fourier'
    window_fn = 'hamming'
    
    causal = True
    mask_nonlinear = 'softmax'
    n_sources = 3
    
    model = GALRNet(N, kernel_size=L, enc_bases=enc_bases, dec_bases=dec_bases, window_fn=window_fn, sep_hidden_channels=H, sep_bottleneck_channels=f, sep_chunk_size=K, sep_hop_size=P, sep_num_blocks=B, causal=causal, sep_norm=sep_norm, mask_nonlinear=mask_nonlinear, n_sources=n_sources)
    print(model)
    print("# Parameters: {}".format(model.num_parameters))
    
    output = model(input)
    print(input.size(), output.size())
    
def _test_galrnet_paper():
    print("Only K and P is different from original, but it doesn't affect the # parameters.")
    batch_size = 2
    K, P = 3, 2
    
    # Encoder & decoder
    C, T = 1, 128
    L, N = 2, 64
    
    # Separator
    f = N
    H = 128 # for each direction
    B = 6
    sep_norm = True
    
    input = torch.randn((batch_size, C, T), dtype=torch.float)
    
    print("-"*10, "Trainable Bases & Non causal", "-"*10)
    enc_bases, dec_bases = 'trainable', 'trainable'
    enc_nonlinear = None
    
    causal = False
    mask_nonlinear = 'sigmoid'
    n_sources = 2
    
    model = GALRNet(N, kernel_size=L, enc_bases=enc_bases, dec_bases=dec_bases, enc_nonlinear=enc_nonlinear, sep_hidden_channels=H, sep_bottleneck_channels=f, sep_chunk_size=K, sep_hop_size=P, sep_num_blocks=B, causal=causal, sep_norm=sep_norm, mask_nonlinear=mask_nonlinear, n_sources=n_sources)
    print(model)
    print("# Parameters: {}".format(model.num_parameters))
    
    output = model(input)
    print(input.size(), output.size())

if __name__ == '__main__':
    print("="*10, "Separator", "="*10)
    _test_separator()
    print()
    
    print("="*10, "GALRNet", "="*10)
    _test_galrnet()
    print()

    print("="*10, "GALRNet (same configuration in paper)", "="*10)
    _test_galrnet_paper()
    print()