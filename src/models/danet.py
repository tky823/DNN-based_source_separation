import torch
import torch.nn as nn

from algorithm.clustering import Kmeans

EPS=1e-12

class DANet(nn.Module):
    def __init__(self, F_bin, embed_dim=20, hidden_channels=600, num_blocks=4, causal=False, mask_nonlinear='sigmoid', iter_clustering=10, eps=EPS, **kwargs):
        super().__init__()
        
        self.F_bin = F_bin
        self.hidden_channels, self.embed_dim = hidden_channels, embed_dim
        self.num_blocks = num_blocks
        
        self.causal = causal
        self.mask_nonlinear = mask_nonlinear
        
        self.lstm = StackedLSTM(F_bin, hidden_channels=hidden_channels, num_blocks=num_blocks, causal=causal)
        self.fc = nn.Linear(hidden_channels, F_bin*embed_dim)
        
        if mask_nonlinear == 'sigmoid':
            self.mask_nonlinear2d = nn.Sigmoid()
        elif mask_nonlinear == 'softmax':
            self.mask_nonlinear2d = nn.Softmax(dim=1)
        else:
            raise NotImplementedError("")
        
        self.iter_clustering = iter_clustering
        self.eps = eps
        
        self.num_parameters = self._get_num_parameters()
    
    def forward(self, input, assignment=None, threshold_weight=None, n_sources=None):
        """
        Args:
            input (batch_size, 1, F_bin, T_bin)
            assignment (batch_size, n_sources, F_bin, T_bin): Speaker assignment when training
            threshold_weight (batch_size, 1, F_bin, T_bin) or <float>
        Returns:
            output (batch_size, n_sources, F_bin, T_bin)
        """
        output, _ = self.extract_latent(input, assignment, threshold_weight=threshold_weight, n_sources=n_sources)
        
        return output
    
    def extract_latent(self, input, assignment=None, threshold_weight=None, n_sources=None):
        """
        input (batch_size, 1, F_bin, T_bin) <torch.Tensor>
        assignment (batch_size, n_sources, F_bin, T_bin) <torch.Tensor>
        threshold_weight (batch_size, 1, F_bin, T_bin) or <float>
        """
        embed_dim = self.embed_dim
        
        if n_sources is not None:
            if assignment is not None and n_sources != assignment.size(1):
                raise ValueError("n_sources is different from assignment.size(1)")
        else:
            n_sources = assignment.size(1)
        
        eps = self.eps
        
        batch_size, _, F_bin, T_bin = input.size()
        
        x = self.lstm(input) # -> (batch_size, T_bin, F_bin)
        x = self.fc(x) # -> (batch_size, T_bin, embed_dim*F_bin)
        x = x.view(batch_size, T_bin, embed_dim, F_bin)
        x = x.permute(0,2,3,1).contiguous()  # -> (batch_size, embed_dim, F_bin, T_bin)
        latent = x.view(batch_size, embed_dim, F_bin*T_bin)
        
        if assignment is None:
            if self.training:
                raise ValueError("assignment is required.")
            latent_kmeans = latent.squeeze(dim=0) # -> (embed_dim, F_bin*T_bin)
            latent_kmeans = latent_kmeans.permute(1,0) # -> (F_bin*T_bin, embed_dim)
            kmeans = Kmeans(latent_kmeans, K=n_sources)
            _, centroids = kmeans(iteration=self.iter_clustering) # (F_bin*T_bin, n_sources), (n_sources, embed_dim)
            attractor = centroids.unsqueeze(dim=0) # (batch_size, n_sources, embed_dim)
        else:
            threshold_weight = threshold_weight.view(batch_size, 1, F_bin*T_bin)
            assignment = assignment.view(batch_size, n_sources, F_bin*T_bin) # -> (batch_size, n_sources, F_bin*T_bin)
            assignment = threshold_weight * assignment
            attractor = torch.bmm(assignment, latent.permute(0,2,1)) / (assignment.sum(dim=2, keepdim=True) + eps) # -> (batch_size, n_sources, embed_dim)
        
        similarity = torch.bmm(attractor, latent) # -> (batch_size, n_sources, F_bin*T_bin)
        similarity = similarity.view(batch_size, n_sources, F_bin, T_bin)
        mask = self.mask_nonlinear2d(similarity) # -> (batch_size, n_sources, F_bin, T_bin)
        output = mask * input
        
        return output, latent
    
    def get_package(self):
        package = {
            'F_bin': self.F_bin,
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
        
        F_bin = package['F_bin']
        embed_dim = package['embed_dim']
        hidden_channels = package['hidden_channels']
        num_blocks = package['num_blocks']
        
        causal = package['causal']
        mask_nonlinear = package['mask_nonlinear']
        
        n_sources = package['n_sources']
        
        eps = package['eps']
        
        model = cls(F_bin, embed_dim=embed_dim, hidden_channels=hidden_channels, num_blocks=num_blocks, causal=causal, mask_nonlinear=mask_nonlinear, n_sources=n_sources, eps=eps)
        
        return model
    
    def _get_num_parameters(self):
        num_parameters = 0
        
        for p in self.parameters():
            if p.requires_grad:
                num_parameters += p.numel()
                
        return num_parameters

class StackedLSTM(nn.Module):
    def __init__(self, F_bin, hidden_channels=600, num_blocks=4, causal=False):
        super().__init__()
        
        net = []
        
        self.num_blocks = num_blocks
        
        if causal:
            bidirectional = False
            num_directions = 1
        else:
            bidirectional = True
            num_directions = 2
        
        for idx in range(num_blocks):
            if idx == 0:
                in_channels = F_bin
                out_channels = hidden_channels//num_directions
            else:
                in_channels = hidden_channels
                out_channels = hidden_channels//num_directions
            
            net.append(nn.LSTM(in_channels, out_channels, batch_first=True, bidirectional=bidirectional))
        
        self.net = nn.Sequential(*net)
        
    def forward(self, input):
        """
        Args:
            input (batch_size, 1, F_bin, T_bin)
        Returns:
            output (batch_size, T_bin, F_bin)
        """
        x = input.squeeze(dim=1).permute(0,2,1) # -> (batch_size, T_bin, F_bin)
        
        for idx in range(self.num_blocks):
            # To avoid error
            x, (_, _) = self.net[idx](x)
        
        output = x

        return output
        
if __name__ == '__main__':
    from algorithm.ideal_mask import ideal_binary_mask
    
    torch.manual_seed(111)
    
    batch_size = 2
    K = 10
    
    H = 32
    B = 4
    
    F_bin, T_bin = 4, 128
    C = 2
    causal = False
    mask_nonlinear = 'sigmoid'
    
    sources = torch.randint(0, 10, (batch_size, C, F_bin, T_bin), dtype=torch.float)
    input = sources.sum(dim=1, keepdim=True)
    assignment = ideal_binary_mask(sources)
    
    model = DANet(F_bin, embed_dim=K, hidden_channels=H, num_blocks=B, causal=causal, mask_nonlinear=mask_nonlinear, n_sources=C)
    print(model)
    output = model(input, assignment)
    
    print(input.size(), output.size())
