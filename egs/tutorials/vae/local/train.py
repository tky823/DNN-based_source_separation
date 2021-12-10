#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

import yaml
import torch
import torchvision

from utils.utils import set_seed
from adhoc_driver import Trainer
from adhoc_utils import choose_vae

parser = argparse.ArgumentParser("Training of VAE")

parser.add_argument("--latent_dim", type=int, default=None, help="Latent dimension")
parser.add_argument("--hidden_channels", type=int, default=None, help="Number of hidden channels")
parser.add_argument("--num_layers", type=int, default=None, help="Number of hidden layers")
parser.add_argument("--num_samples", type=int, default=None, help="Number of samples")
parser.add_argument('--optimizer', type=str, default='adam', choices=['sgd', 'adam', 'rmsprop'], help='Optimizer, [sgd, adam, rmsprop]')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate. Default: 1e-3')
parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay (L2 penalty). Default: 0')
parser.add_argument('--max_norm', type=float, default=None, help='Gradient clipping')
parser.add_argument('--batch_size', type=int, default=100, help='Batch size. Default: 100')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
parser.add_argument('--model_dir', type=str, default='./tmp/model', help='Model directory')
parser.add_argument('--loss_dir', type=str, default='./tmp/loss', help='Loss directory')
parser.add_argument('--sample_dir', type=str, default='./tmp/sample', help='Sample directory')
parser.add_argument('--continue_from', type=str, default=None, help='Resume training')
parser.add_argument('--use_cuda', type=int, default=1, help='0: Not use cuda, 1: Use cuda')
parser.add_argument('--overwrite', type=int, default=0, help='0: NOT overwrite, 1: FORCE overwrite')
parser.add_argument('--seed', type=int, default=42, help='Random seed')

def main(args):
    set_seed(args.seed)
    
    args.in_channels = 28 * 28
    
    # Preprocessing
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda x: x.view(-1))
    ])

    train_dataset = torchvision.datasets.MNIST("./data/MNIST", train=True, download=True, transform=transform)
    valid_dataset = torchvision.datasets.MNIST("./data/MNIST", train=False, download=True, transform=transform)

    loader = {}
    loader['train'] = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    loader['valid'] = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

    with open(args.config_path) as f:
        kwargs = yaml.safe_load(f)
    model = choose_vae(**kwargs)
    print(model)
    
    if torch.cuda.is_available():
        model.cuda()
        print("Use CUDA")
    else:
        print("Not use CUDA")
        
    # Optimizer
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise ValueError("Not support optimizer {}".format(args.optimizer))

    if args.max_norm is not None and args.max_norm == 0:
        args.max_norm = None
    
    trainer = Trainer(model, loader, optimizer, args)
    trainer.run()

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    
    main(args)
