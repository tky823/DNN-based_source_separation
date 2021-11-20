import torch
import torch.nn as nn
import torch.nn.functional as F

EPS=1e-12

class DeepEmbedding(nn.Module):
    def __init__(self, n_bins, hidden_channels=300, embed_dim=40, num_layers=2, causal=False, eps=EPS, **kwargs):
        super().__init__()

        self.n_bins = n_bins
        self.hidden_channels, self.embed_dim = hidden_channels, embed_dim

        self.eps = eps

        if causal:
            bidirectional = False
            num_directions = 1
        else:
            bidirectional = True
            num_directions = 2

        self.rnn = nn.LSTM(n_bins, hidden_channels, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Linear(num_directions*hidden_channels, n_bins*embed_dim)

        self.num_parameters = self._get_num_parameters()
    
    def forward(self, input):
        """
        Args:
            input (batch_size, n_bins, n_frames): Input feature. This input is expected log-magnitude.
        Returns:
            output (batch_size, embed_dim, n_bins, n_frames): Embedded feature.
        """
        n_bins, embed_dim = self.n_bins, self.embed_dim
        eps = self.eps

        batch_size, _, n_frames = input.size()

        x = input.permute(0,2,1).contiguous() # (batch_size, n_frames, n_bins)
        x, (_, _) = self.rnn(x)
        x = self.fc(x) # (batch_size, n_frames, n_bins*embed_dim)
        x = x.view(batch_size, n_frames, n_bins, embed_dim)
        x = x.permute(0,3,2,1).contiguous() # (batch_size, embed_dim, n_bins, n_frames)
        norm = torch.sum(x**2, dim=1, keepdim=True)
        output = x / (norm + eps)

        return output
    
    def _get_num_parameters(self):
        num_parameters = 0
        
        for p in self.parameters():
            if p.requires_grad:
                num_parameters += p.numel()
                
        return num_parameters

class DeepEmbedding_pp(nn.Module):
    def __init__(self, n_bins, hidden_channels=300, embed_dim=40, num_layers=4, enh_hidden_channels=600, enh_num_layers=2, causal=False, eps=EPS, **kwargs):
        super().__init__()

        self.n_bins = n_bins
        self.hidden_channels, self.embed_dim = hidden_channels, embed_dim

        self.eps = eps

        if causal:
            bidirectional = False
            num_directions = 1
        else:
            bidirectional = True
            num_directions = 2

        self.rnn = nn.LSTM(n_bins, hidden_channels, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_channels*num_directions, n_bins*embed_dim)
        self.embed_nonlinear = nn.Sigmoid()
        self.net_enhancement = NaiveEnhancementNet(2*n_bins, n_bins, hidden_channels=enh_hidden_channels, num_layers=enh_num_layers, causal=causal, eps=eps)

        self.num_parameters = self._get_num_parameters()
    
    def forward(self, input):
        """
        Args:
            input (batch_size, n_bins, n_frames): Input feature. This input is expected log-magnitude.
        Returns:
            output (batch_size, embed_dim, n_bins, n_frames): Embedded feature.
        """
        n_bins, embed_dim = self.n_bins, self.embed_dim
        eps = self.eps

        batch_size, _, n_frames = input.size()

        x = input.permute(0,2,1).contiguous() # (batch_size, n_frames, n_bins)
        x, (_, _) = self.rnn(x)
        x = self.fc(x) # (batch_size, n_frames, n_bins*embed_dim)
        x = x.view(batch_size, n_frames, n_bins, embed_dim)
        x = x.permute(0,3,2,1).contiguous() # (batch_size, embed_dim, n_bins, n_frames)
        norm = torch.sum(x**2, dim=1, keepdim=True)
        x = x / (norm + eps)
        output = self.embed_nonlinear(x)

        return output
    
    def _get_num_parameters(self):
        num_parameters = 0
        
        for p in self.parameters():
            if p.requires_grad:
                num_parameters += p.numel()
                
        return num_parameters

class NaiveEnhancementNet(nn.Module):
    def __init__(self, num_features, n_bins, hidden_channels=300, num_layers=2, causal=False, eps=EPS, **kwargs):
        super().__init__()

        self.eps = eps

        if causal:
            bidirectional = False
            num_directions = 1
        else:
            bidirectional = True
            num_directions = 2
        
        self.rnn = nn.LSTM(num_features, hidden_channels, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_channels*num_directions, n_bins)
        self.nonlinear = nn.Softmax(dim=1)
    
    def forward(self, input):
        return input



class ChimeraNet(nn.Module):
    def __init__(self, n_bins, hidden_channels=300, embed_dim=20, num_layers=2, causal=False, n_sources=2, eps=EPS, **kwargs):
        super().__init__()

        self.n_bins = n_bins
        self.hidden_channels, self.embed_dim = hidden_channels, embed_dim

        self.eps = eps

        if causal:
            bidirectional = True
            num_directions = 2
        else:
            bidirectional = False
            num_directions = 1

        self.rnn = nn.LSTM(n_bins, hidden_channels, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)

        self.embed_fc = nn.Linear(hidden_channels*num_directions, n_bins*embed_dim)
        self.embed_nonlinear = nn.Tanh()

        self.mask_fc = nn.Linear(hidden_channels, n_bins*n_sources)
        self.mask_nonlinear = nn.Softmax(dim=1)
        

        self.num_parameters = self._get_num_parameters()
    
    def forward(self, input):
        """
        Args:
            input (batch_size, n_bins, n_frames): Input feature. This input is expected log-magnitude.
        Returns:
            output (batch_size, embed_dim, n_bins, n_frames): Embedded feature.
        """
        n_bins, embed_dim = self.n_bins, self.embed_dim
        eps = self.eps

        batch_size, _, n_frames = input.size()

        x = input.permute(0,2,1).contiguous() # (batch_size, n_frames, n_bins)
        x, (_, _) = self.rnn(x)
        x = self.fc(x) # (batch_size, n_frames, n_bins*embed_dim)
        x = x.view(batch_size, n_frames, n_bins, embed_dim)
        x = x.permute(0,3,2,1).contiguous() # (batch_size, embed_dim, n_bins, n_frames)
        norm = torch.sum(x**2, dim=1, keepdim=True)
        output = x / (norm + eps)

        return output
    
    def _get_num_parameters(self):
        num_parameters = 0
        
        for p in self.parameters():
            if p.requires_grad:
                num_parameters += p.numel()
                
        return num_parameters

def _test_deep_embedding():
    batch_size, T = 2, 512
    n_sources = 2
    n_fft, hop_length = 256, 128
    window_fn = 'hann'
    n_bins = n_fft//2 + 1
    hidden_channels, embed_dim = 300, 40

    stft = BatchSTFT(n_fft=n_fft, hop_length=hop_length, window_fn=window_fn)
    istft = BatchInvSTFT(n_fft=n_fft, hop_length=hop_length, window_fn=window_fn)
    criterion = AffinityLoss()

    signal = torch.randn((batch_size*n_sources, T), dtype=torch.float)
    spectrogram = stft(signal)
    real, imag = spectrogram[...,0], spectrogram[...,1]
    power = real**2+imag**2
    target = 10 * torch.log10(power + EPS)
    _, _, n_frames = target.size()
    target = target.view(batch_size, n_sources, n_bins, n_frames)
    target = ideal_binary_mask(target)
    input = target.sum(dim=1)

    # Non causal
    print("-"*10, "Non causal", "-"*10)

    model = DeepEmbedding_pp(n_bins, hidden_channels, embed_dim=embed_dim, causal=False)
    print(model)
    print("# Parameters: {}".format(model.num_parameters))
    
    output = model(input)
    print(input.size(), output.size())
    
    loss = criterion(output, target)
    print(loss.item())

def _test_chimeranet():
    pass

if __name__ == '__main__':
    from algorithm.stft import BatchSTFT, BatchInvSTFT
    from algorithm.frequency_mask import ideal_binary_mask
    from criterion.deep_clustering import AffinityLoss

    torch.manual_seed(111)
    
    print("="*10, "Deep embedding", "="*10)
    _test_deep_embedding()
    print()

    print("="*10, "Chimera Net", "="*10)
    _test_chimeranet()