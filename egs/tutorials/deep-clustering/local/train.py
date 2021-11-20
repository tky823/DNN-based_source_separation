#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import torch
import torch.nn as nn

from utils.utils import set_seed
from dataset import SpectrogramTrainDataset, TrainDataLoader
from driver import Trainer
from models.deep_clustering import DeepEmbedding
from criterion.sdr import AffinityLoss

parser = argparse.ArgumentParser(description="Training of DeepClustering")

parser.add_argument('--train_json_path', type=str, default=None, help='Path for train.json')
parser.add_argument('--valid_json_path', type=str, default=None, help='Path for valid.json')
parser.add_argument('--sample_rate', '-sr', type=int, default=16000, help='Sampling rate')
parser.add_argument('--n_sources', type=int, default=2, help='Number of sources')
parser.add_argument('--samples', type=int, default=16384, help='Number of input samples')
parser.add_argument('--dimension', '-D', type=int, default=20, help='Number of dimensions on embedding')
parser.add_argument('--n_bins', '-F', type=int, default=256, help='Number of frequency bins')
parser.add_argument('--hop_length', '-S', type=int, default=256, help='Hop length of STFT')
parser.add_argument('--hidden_channels', '-H', type=int, default=256, help='Number of hidden channels')
parser.add_argument('--num_layers', '-R', type=int, default=3, help='Number of layers of LSTM')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
parser.add_argument('--epochs', type=int, default=100, help='Epoch')
parser.add_argument('--causal', type=int, default=1, help='Causality')
parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
parser.add_argument('--criterion', type=str, default='affinity', choices=['affinity'], help='Criterion')
parser.add_argument('--exp_dir', type=str, default='./tmp', help='Path to experiment')
parser.add_argument('--continue_from', type=str, default=None, help='Model path when resuming training')

def main(args):
    set_seed(args.seed)
    
    train_dataset = SpectrogramTrainDataset(args.wav_root, args.train_json_path)
    valid_dataset = SpectrogramTrainDataset(args.wav_root, args.valid_json_path)
    print("Training dataset includes {} samples.".format(len(train_dataset)))
    print("Valid dataset includes {} samples.".format(len(valid_dataset)))
    
    loader = {}
    loader['train'] = TrainDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    loader['valid'] = TrainDataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

    args.domain = 'amplitude'
    args.conditioning = None
    args.fft_size = (args.n_bins - 1) * 2
    
    win_length = args.win_length
    
    if args.hop_length is None:
        hop_length = win_length//4
    else:
        hop_length = args.hop_size
    if args.max_norm is not None and args.max_norm == 0:
        args.max_norm = None
    model = DeepEmbedding(args.n_bins, hidden_channels=args.hidden_channels, num_layers=args.num_layers, dimension=args.dimension, num_clusters=args.n_sources, causal=args.causal)
    print(model)
    print("# Parameters: {}".format(model.num_parameters))
    
    if args.use_cuda:
        if torch.cuda.is_available():
            model.cuda()
            model = nn.DataParallel(model)
            print("Use CUDA")
        else:
            raise ValueError("Cannot use CUDA.")
    else:
        print("Does NOT use CUDA")
    
    # Optimizer
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise ValueError("Not support optimizer {}".format(args.optimizer))
    
    # Criterion
    if args.criterion == 'affinity':
        criterion = AffinityLoss()
    else:
        raise ValueError("Not support criterion {}".format(args.criterion))
    
    trainer = Trainer(model, loader, criterion, optimizer, args)
    trainer.run()

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    main(args)
    
    
    
