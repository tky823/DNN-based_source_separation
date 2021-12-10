from models.vae import Encoder as NaiveEncoder
from models.vae import Decoder as NaiveDecoder
from models.vae import VAE, NormalLatentSampler

def choose_vae(name, **kwargs):
    if name == "naive-normal":
        in_channels, hidden_channels = kwargs["in_channels"], kwargs["hidden_channels"]
        latent_dim = kwargs["latent_dim"]
        num_layers = kwargs["num_layers"]
        encoder = NaiveEncoder(in_channels, hidden_channels, latent_dim=latent_dim, num_layers=num_layers)
        decoder = NaiveDecoder(in_channels, hidden_channels, latent_dim=latent_dim, num_layers=num_layers)
        latent_sampler = NormalLatentSampler()
    else:
        raise ValueError("Not support model {}".format(name))
    
    model = VAE(encoder, decoder, latent_sampler=latent_sampler)

    return model