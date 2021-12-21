#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

import yaml
import torch
import torch.nn as nn

from utils.utils import set_seed
from dataset import IdealMaskSpectrogramTrainDataset, IdealMaskSpectrogramEvalDataset, TrainDataLoader, EvalDataLoader
from adhoc_driver import AdhocTrainer
from models.adanet import ADANet
from adhoc_criterion import SquaredError

parser = argparse.ArgumentParser(description="Training of ADANet")

parser.add_argument('--train_wav_root', type=str, default=None, help='Path for training dataset ROOT directory')
parser.add_argument('--valid_wav_root', type=str, default=None, help='Path for validation dataset ROOT directory')
parser.add_argument('--train_list_path', type=str, default=None, help='Path for mix_<n_sources>_spk_<max,min>_tr_mix')
parser.add_argument('--valid_list_path', type=str, default=None, help='Path for mix_<n_sources>_spk_<max,min>_cv_mix')
parser.add_argument('--sample_rate', '-sr', type=int, default=8000, help='Sampling rate')
parser.add_argument('--duration', type=float, default=2, help='Duration')
parser.add_argument('--valid_duration', type=float, default=0, help='Duration for valid dataset for avoiding memory error.')
parser.add_argument('--window_fn', type=str, default='hamming', help='Window function')
parser.add_argument('--ideal_mask', type=str, default='ibm', choices=['ibm', 'irm', 'wfm'], help='Ideal mask for assignment')
parser.add_argument('--threshold', type=float, default=40, help='Wight threshold. Default: 40 ')
parser.add_argument('--target_type', type=str, default='source', choices=['source', 'oracle'], help='Target type DNN tries to output.')
parser.add_argument('--n_fft', type=int, default=256, help='Window length')
parser.add_argument('--hop_length', type=int, default=None, help='Hop size')
parser.add_argument('--embed_dim', '-K', type=int, default=20, help='Embedding dimension')
parser.add_argument('--hidden_channels', '-H', type=int, default=300, help='hidden_channels')
parser.add_argument('--num_blocks', '-B', type=int, default=4, help='# LSTM layers')
parser.add_argument('--num_anchors', '-N', type=int, default=4, help='# of anchors')
parser.add_argument('--dropout', type=float, default=0, help='Dropout rate of LSTM layers')
parser.add_argument('--causal', type=int, default=0, help='Causality')
parser.add_argument('--mask_nonlinear', type=str, default='sigmoid', help='Non-linear function of mask estiamtion')
parser.add_argument('--take_log', type=int, default=1, help='Whether to apply log for input.')
parser.add_argument('--take_db', type=int, default=0, help='Whether to apply 20*log10 for input.')
parser.add_argument('--n_sources', type=int, default=None, help='# speakers')
parser.add_argument('--criterion', type=str, default='se', choices=['se', 'l1loss', 'l2loss'], help='Criterion')
parser.add_argument('--optimizer', type=str, default='rmsprop', choices=['sgd', 'adam', 'rmsprop'], help='Optimizer, [sgd, adam, rmsprop]')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate. Default: 1e-4')
parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay (L2 penalty). Default: 0')
parser.add_argument('--max_norm', type=float, default=None, help='Gradient clipping')
parser.add_argument('--scheduler_path', type=str, default=None, help='Path to scheduler.yaml')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size. Default: 64')
parser.add_argument('--epochs', type=int, default=150, help='Number of epochs')
parser.add_argument('--model_dir', type=str, default='./tmp/model', help='Model directory')
parser.add_argument('--loss_dir', type=str, default='./tmp/loss', help='Loss directory')
parser.add_argument('--sample_dir', type=str, default='./tmp/sample', help='Sample directory')
parser.add_argument('--continue_from', type=str, default=None, help='Resume training')
parser.add_argument('--use_cuda', type=int, default=1, help='0: Not use cuda, 1: Use cuda')
parser.add_argument('--overwrite', type=int, default=0, help='0: NOT overwrite, 1: FORCE overwrite')
parser.add_argument('--seed', type=int, default=42, help='Random seed')

def main(args):
    set_seed(args.seed)

    samples = int(args.sample_rate * args.duration)
    overlap = 0

    train_dataset = IdealMaskSpectrogramTrainDataset(args.train_wav_root, args.train_list_path, n_fft=args.n_fft, hop_length=args.hop_length, window_fn=args.window_fn, mask_type=args.ideal_mask, threshold=args.threshold, samples=samples, overlap=overlap, n_sources=args.n_sources)

    max_samples = int(args.sample_rate * args.valid_duration)
    valid_dataset = IdealMaskSpectrogramEvalDataset(args.valid_wav_root, args.valid_list_path, n_fft=args.n_fft, hop_length=args.hop_length, window_fn=args.window_fn, mask_type=args.ideal_mask, threshold=args.threshold, max_samples=max_samples, n_sources=args.n_sources)

    print("Training dataset includes {} samples.".format(len(train_dataset)))
    print("Valid dataset includes {} samples.".format(len(valid_dataset)))

    loader = {}
    loader['train'] = TrainDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    loader['valid'] = EvalDataLoader(valid_dataset, batch_size=1, shuffle=False)

    if args.max_norm is not None and args.max_norm == 0:
        args.max_norm = None

    args.n_bins = args.n_fft // 2 + 1
    model = ADANet(args.n_bins, embed_dim=args.embed_dim, hidden_channels=args.hidden_channels, num_blocks=args.num_blocks, num_anchors=args.num_anchors, dropout=args.dropout, causal=args.causal, mask_nonlinear=args.mask_nonlinear, take_log=args.take_log, take_db=args.take_db)
    print(model)
    print("# Parameters: {}".format(model.num_parameters))

    if args.use_cuda:
        if torch.cuda.is_available():
            model.cuda()
            model = nn.DataParallel(model)
            print("Use CUDA", flush=True)
        else:
            raise ValueError("Cannot use CUDA.")
    else:
        print("Does NOT use CUDA", flush=True)

    # Optimizer
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise ValueError("Not support optimizer {}".format(args.optimizer))

    # Scheduler
    with open(args.scheduler_path) as f:
        config_scheduler = yaml.safe_load(f)

    if config_scheduler['scheduler'] == 'ReduceLROnPlateau':
        config_scheduler.pop('scheduler')
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **config_scheduler)
    elif config_scheduler['scheduler'] == 'ExponentialLR':
        config_scheduler.pop('scheduler')
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, **config_scheduler)
    else:
        raise NotImplementedError("Not support schduler {}.".format(args.scheduler))

    # Criterion
    if args.criterion == 'se':
        criterion = SquaredError(sum_dim=2, mean_dim=(1,3)) # (batch_size, n_sources, n_bins, n_frames)
    else:
        raise ValueError("Not support criterion {}".format(args.criterion))

    trainer = AdhocTrainer(model, loader, criterion, optimizer, scheduler, args)
    trainer.run()

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    main(args)
