import matplotlib.pyplot as plt
import torch

from models.vae import VAE
from models.vae import Encoder, Decoder, NormalLatentSampler

def draw_loss_curve(train_loss, valid_loss=None, save_path="./loss.png"):
    plt.figure()

    epochs = range(1, len(train_loss) + 1)

    if isinstance(train_loss, torch.Tensor):
        train_loss = train_loss.numpy()

    plt.plot(epochs, train_loss, label="train")

    if valid_loss is not None:
        if isinstance(valid_loss, torch.Tensor):
            valid_loss = valid_loss.numpy()
        plt.plot(epochs, valid_loss, label="valid")

    plt.yscale("log")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

def choose_vae(name, in_channels, hidden_channels, latent_dim, num_layers=3):
    if name == "naive-normal":
        encoder = Encoder(in_channels, hidden_channels, latent_dim=latent_dim, num_layers=num_layers)
        decoder = Decoder(in_channels, hidden_channels, latent_dim=latent_dim, num_layers=num_layers)
        latent_sampler = NormalLatentSampler()
    else:
        raise NotImplementedError

    model = VAE(encoder, decoder, latent_sampler=latent_sampler)

    return model