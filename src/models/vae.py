import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, latent_sampler: nn.Module):
        super().__init__()

        self.encoder = encoder
        self.latent_sampler = latent_sampler
        self.decoder = decoder
        
    def forward(self, input, num_samples=1, return_params=False):
        """
        Args:
            input: (batch_size, *)
        Returns:
            output: (batch_size, num_samples, *)
        """
        outputs = self.extract_latent(input, num_samples=num_samples)
        output, latent = outputs[:2]
        params = output[2:]
        
        if return_params:
            return output, latent, *params
        
        return output
    
    def extract_latent(self, input, num_samples=1):
        """
        Args:
            input: (batch_size, *)
        Returns:
            output: (batch_size, num_samples, *)
        """
        params = self.encoder(input)
        
        if type(params) is not tuple:
            params = (params,)
        
        latent = self.latent_sampler(*params, num_samples=num_samples) # latent: (batch_size, num_samples, latent_dim)
        output = self.decoder(latent)
        
        return output, latent, *params
        
class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, latent_dim, num_layers=3):
        super().__init__()

        net = []
        
        for n in range(num_layers - 1):
            if n == 0:
                net.append(nn.Linear(in_channels, hidden_channels))
            else:
                net.append(nn.Linear(hidden_channels, hidden_channels))
            net.append(nn.ReLU())

        self.net = nn.Sequential(*net)
        self.linear_mean = nn.Linear(hidden_channels, latent_dim)
        self.linear_var = nn.Linear(hidden_channels, latent_dim)
        self.activation_var = nn.Softplus()
        
    def forward(self, input):
        x = self.net(input)
    
        output_mean = self.linear_mean(x)
        x_var = self.linear_var(x)
        output_var = self.activation_var(x_var)
        
        return output_mean, output_var
        
class Decoder(nn.Module):
    def __init__(self, out_channels, hidden_channels, latent_dim, num_layers=3):
        super().__init__()
        
        net = []
        
        for n in range(num_layers):
            if n == 0:
                net.append(nn.Linear(latent_dim, hidden_channels))
            elif n == num_layers - 1:
                net.append(nn.Linear(hidden_channels, out_channels))
            else:
                net.append(nn.Linear(hidden_channels, hidden_channels))
            if n == num_layers-1:
                net.append(nn.Sigmoid())
            else:
                net.append(nn.ReLU())
        self.net = nn.Sequential(*net)
        
    def forward(self, latent):
        output = self.net(latent)
        return output
        
class NormalLatentSampler(nn.Module):
    def __init__(self):
        super().__init__()

        loc, scale = 0, 1
        self.backend_sampler = torch.distributions.normal.Normal(loc, scale=scale)
    
    def forward(self, mean, var, num_samples=1):
        """
        Args:
            mean: (batch_size, latent_dim)
            var: (batch_size, latent_dim)
        Returns:
            latent: (batch_size, num_samples, latent_dim)
        """
        batch_size, latent_dim = mean.size()
        mean, var = mean.unsqueeze(dim=1), var.unsqueeze(dim=1)

        sample_shape = (batch_size, num_samples, latent_dim)
        epsilon = self.backend_sampler.sample(sample_shape)
        latent = mean + torch.sqrt(var) * epsilon
            
        return latent

if __name__ == '__main__':
    batch_size = 4
    latent_dim = 10
    num_layers = 4
    in_channels, hidden_channels = 28*28, 200
    size_input = (batch_size, in_channels)

    encoder = Encoder(in_channels, hidden_channels, latent_dim=latent_dim, num_layers=num_layers)
    decoder = Decoder(in_channels, hidden_channels, latent_dim=latent_dim, num_layers=num_layers)
    latent_sampler = NormalLatentSampler()
    model = VAE(encoder, decoder, latent_sampler=latent_sampler)
    
    input = torch.randint(0, 256, size_input) / 256
    output = model(input, return_params=False)
    
    print(model)
    print(input.size(), output.size())