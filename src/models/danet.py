import torch
import torch.nn as nn

EPS=1e-12

class DANet(nn.Module):
    def __init__(self, F_bin, embed_dim=20, hidden_channels=600, num_blocks=4, causal=False, mask_nonlinear='sigmoid', n_sources=2, eps=EPS, **kwargs):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_blocks = num_blocks
        
        self.lstm = StackedLSTM(F_bin, hidden_channels=hidden_channels, num_blocks=num_blocks, causal=causal)
        self.fc = nn.Linear(hidden_channels, F_bin*embed_dim)
        
        if mask_nonlinear == 'sigmoid':
            self.mask_nonlinear = nn.Sigmoid()
        elif mask_nonlinear == 'softmax':
            self.mask_nonlinear = nn.Softmax(dim=1)
        else:
            raise NotImplementedError("")
    
        self.n_sources = n_sources
        
        self.num_parameters = self._get_num_parameters()
    
    def forward(self, input, assignment, threshold=None):
        """
        Args:
            input (batch_size, 1, F_bin, T_bin)
            assignment (batch_size, n_sources, F_bin, T_bin): Speaker assignment when training
        Returns:
            output (batch_size, n_sources, F_bin, T_bin)
        """
        embed_dim = self.embed_dim
        n_sources = self.n_sources
        
        batch_size, _, F_bin, T_bin = input.size()
        
        x = self.lstm(input) # -> (batch_size, T_bin, F_bin)
        x = self.fc(x) # -> (batch_size, T_bin, embed_dim*F_bin)
        x = x.view(batch_size, T_bin, embed_dim, F_bin)
        x = x.permute(0,2,3,1).contiguous()  # -> (batch_size, embed_dim, F_bin, T_bin)
        x = x.view(batch_size, embed_dim, F_bin*T_bin)
        
        input = input.view(batch_size, 1, F_bin*T_bin)
        assignment = assignment.view(batch_size, n_sources, F_bin*T_bin) # -> (batch_size, n_sources, F_bin*T_bin)
        
        if not self.training:
            if threshold is None:
                raise ValueError("Specify threshold!")
            
            w = torch.where(input>threshold, torch.ones_like(input), torch.zeros_like(input))  # -> (batch_size, 1, F_bin*T_bin)
            assignment = w * assignment
        
        attractor = torch.bmm(assignment, x.permute(0,2,1)) / assignment.sum(dim=2, keepdim=True) # -> (batch_size, n_sources, K)
        
        similarity = torch.bmm(attractor, x) # -> (batch_size, n_sources, F_bin*T_bin)
        similarity = similarity.view(batch_size, n_sources, F_bin, T_bin)
        mask = self.mask_nonlinear(similarity) # -> (batch_size, n_sources, F_bin, T_bin)
        output = mask * input
        
        return output
    
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
