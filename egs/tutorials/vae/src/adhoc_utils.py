from models.vae import VAE
from models.vae import Encoder, Decoder, NormalLatentSampler

def choose_vae(name, in_channels, hidden_channels, latent_dim, num_layers=3):
    if name == "naive-normal":
        encoder = Encoder(in_channels, hidden_channels, latent_dim=latent_dim, num_layers=num_layers)
        decoder = Decoder(in_channels, hidden_channels, latent_dim=latent_dim, num_layers=num_layers)
        latent_sampler = NormalLatentSampler()
    else:
        raise NotImplementedError

    model = VAE(encoder, decoder, latent_sampler=latent_sampler)

    return model