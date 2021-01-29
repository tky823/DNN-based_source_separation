import torch
import torch.nn as nn

from algorithm.clustering import Kmeans

EPS=1e-12

class DANet(nn.Module):
    def __init__(self, n_bins, embed_dim=20, hidden_channels=600, num_blocks=4, causal=False, mask_nonlinear='sigmoid', iter_clustering=10, eps=EPS, **kwargs):
        super().__init__()
        
        self.n_bins = n_bins
        self.hidden_channels, self.embed_dim = hidden_channels, embed_dim
        self.num_blocks = num_blocks

        self.causal = causal

        if causal:
            num_directions = 1
            bidirectional = False
        else:
            num_directions = 2
            bidirectional = True
        
        self.mask_nonlinear = mask_nonlinear
        
        # self.lstm = StackedLSTM(n_bins, hidden_channels=hidden_channels, num_blocks=num_blocks, causal=causal)
        self.lstm = nn.LSTM(n_bins, hidden_channels, num_layers=num_blocks, batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Linear(num_directions*hidden_channels, n_bins*embed_dim)
        
        if mask_nonlinear == 'sigmoid':
            self.mask_nonlinear2d = nn.Sigmoid()
        elif mask_nonlinear == 'softmax':
            self.mask_nonlinear2d = nn.Softmax(dim=1)
        else:
            raise NotImplementedError("")
        
        self.iter_clustering = iter_clustering
        self.eps = eps
        
        self.num_parameters = self._get_num_parameters()
    
    def forward(self, input, assignment=None, threshold_weight=None, n_sources=None, iter_clustering=None):
        """
        Args:
            input (batch_size, 1, n_bins, n_frames): Amplitude
            assignment (batch_size, n_sources, n_bins, n_frames): Speaker assignment when training
            threshold_weight (batch_size, 1, n_bins, n_frames) or <float>
        Returns:
            output (batch_size, n_sources, n_bins, n_frames)
        """
        output, _ = self.extract_latent(input, assignment, threshold_weight=threshold_weight, n_sources=n_sources, iter_clustering=None)
        
        return output
    
    def extract_latent(self, input, assignment=None, threshold_weight=None, n_sources=None, iter_clustering=None):
        """
        Args:
            input (batch_size, 1, n_bins, n_frames) <torch.Tensor>
            assignment (batch_size, n_sources, n_bins, n_frames) <torch.Tensor>
            threshold_weight (batch_size, 1, n_bins, n_frames) or <float>
        """
        if iter_clustering is None:
            iter_clustering = self.iter_clustering
        
        if n_sources is not None:
            if assignment is not None and n_sources != assignment.size(1):
                raise ValueError("n_sources is different from assignment.size(1)")
        else:
            if assignment is None:
                raise ValueError("Specify assignment, given None!")
            n_sources = assignment.size(1)
        
        embed_dim = self.embed_dim
        eps = self.eps
        
        batch_size, _, n_bins, n_frames = input.size()
        
        log_amplitude = torch.log(input + eps)
        x = log_amplitude.squeeze(dim=1).permute(0,2,1) # -> (batch_size, n_frames, n_bins)
        x, (_, _) = self.lstm(x) # -> (batch_size, n_frames, n_bins)
        x = self.fc(x) # -> (batch_size, n_frames, embed_dim*n_bins)
        x = x.view(batch_size, n_frames, embed_dim, n_bins)
        x = x.permute(0,2,3,1).contiguous()  # -> (batch_size, embed_dim, n_bins, n_frames)
        latent = x.view(batch_size, embed_dim, n_bins*n_frames)
        
        if assignment is None:
            # TODO: test threshold
            if self.training:
                raise ValueError("assignment is required.")
            latent_kmeans = latent.squeeze(dim=0) # -> (embed_dim, n_bins*n_frames)
            latent_kmeans = latent_kmeans.permute(1,0) # -> (n_bins*n_frames, embed_dim)
            kmeans = Kmeans(latent_kmeans, K=n_sources)
            _, centroids = kmeans(iteration=iter_clustering) # (n_bins*n_frames, n_sources), (n_sources, embed_dim)
            attractor = centroids.unsqueeze(dim=0) # (batch_size, n_sources, embed_dim)
        else:
            threshold_weight = threshold_weight.view(batch_size, 1, n_bins*n_frames)
            assignment = assignment.view(batch_size, n_sources, n_bins*n_frames) # -> (batch_size, n_sources, n_bins*n_frames)
            assignment = threshold_weight * assignment
            attractor = torch.bmm(assignment, latent.permute(0,2,1)) / (assignment.sum(dim=2, keepdim=True) + eps) # -> (batch_size, n_sources, embed_dim)
        
        similarity = torch.bmm(attractor, latent) # -> (batch_size, n_sources, n_bins*n_frames)
        similarity = similarity.view(batch_size, n_sources, n_bins, n_frames)
        mask = self.mask_nonlinear2d(similarity) # -> (batch_size, n_sources, n_bins, n_frames)
        output = mask * input
        
        return output, latent
    
    def get_package(self):
        package = {
            'n_bins': self.n_bins,
            'embed_dim': self.embed_dim,
            'hidden_channels': self.hidden_channels,
            'num_blocks': self.num_blocks,
            'causal': self.causal,
            'mask_nonlinear': self.mask_nonlinear,
            'eps': self.eps
        }
        
        return package
    
    @classmethod
    def build_model(cls, model_path):
        package = torch.load(model_path, map_location=lambda storage, loc: storage)
        
        n_bins = package['n_bins']
        embed_dim = package['embed_dim']
        hidden_channels = package['hidden_channels']
        num_blocks = package['num_blocks']
        
        causal = package['causal']
        mask_nonlinear = package['mask_nonlinear']
        
        eps = package['eps']
        
        model = cls(n_bins, embed_dim=embed_dim, hidden_channels=hidden_channels, num_blocks=num_blocks, causal=causal, mask_nonlinear=mask_nonlinear, eps=eps)
        
        return model
    
    def _get_num_parameters(self):
        num_parameters = 0
        
        for p in self.parameters():
            if p.requires_grad:
                num_parameters += p.numel()
                
        return num_parameters

def _test_danet():
    torch.manual_seed(111)

    batch_size = 2
    K = 10
    
    H = 32
    B = 4
    
    n_bins, n_frames = 4, 128
    C = 2
    causal = False
    mask_nonlinear = 'sigmoid'
    
    sources = torch.randn((batch_size, C, n_bins, n_frames), dtype=torch.float)
    input = sources.sum(dim=1, keepdim=True)
    assignment = ideal_binary_mask(sources)
    threshold_weight = torch.randint(0, 2, (batch_size, 1, n_bins, n_frames), dtype=torch.float)
    
    model = DANet(n_bins, embed_dim=K, hidden_channels=H, num_blocks=B, causal=causal, mask_nonlinear=mask_nonlinear, n_sources=C)
    print(model)
    print("# Parameters: {}".format(model.num_parameters))

    output = model(input, assignment, threshold_weight=threshold_weight)
    
    print(input.size(), output.size())

def _test_danet_paper():
    torch.manual_seed(111)

    batch_size = 2
    K = 20
    
    H = 600
    B = 4
    
    n_bins, n_frames = 4, 128
    C = 2
    causal = False
    mask_nonlinear = 'sigmoid'
    
    sources = torch.randn((batch_size, C, n_bins, n_frames), dtype=torch.float)
    input = sources.sum(dim=1, keepdim=True)
    assignment = ideal_binary_mask(sources)
    threshold_weight = torch.randint(0, 2, (batch_size, 1, n_bins, n_frames), dtype=torch.float)
    
    model = DANet(n_bins, embed_dim=K, hidden_channels=H, num_blocks=B, causal=causal, mask_nonlinear=mask_nonlinear, n_sources=C)
    print(model)
    print("# Parameters: {}".format(model.num_parameters))

    output = model(input, assignment, threshold_weight=threshold_weight)
    
    print(input.size(), output.size())
        
if __name__ == '__main__':
    from algorithm.frequency_mask import ideal_binary_mask

    print("="*10, "DANet", "="*10)
    _test_danet()
    print()

    print("="*10, "DANet (same configuration in paper)", "="*10)
    _test_danet_paper()
    print()