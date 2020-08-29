import random
import numpy as np
import matplotlib.pyplot as plt
import torch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def draw_loss(train_loss, valid_loss=None, save_path='./loss.png'):
    plt.figure()

    epochs = range(1, len(train_loss) + 1)

    if isinstance(train_loss, torch.Tensor):
        train_loss = train_loss.numpy()

    plt.plot(epochs, train_loss, label='train')

    if valid_loss is not None:
        if isinstance(valid_loss, torch.Tensor):
            valid_loss = valid_loss.numpy()
        plt.plot(epochs, valid_loss, label='valid')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
