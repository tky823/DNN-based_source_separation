#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

import torch
import torch.nn as nn

from utils.utils import set_seed
from adhoc_utils import create_spk_to_idx
from adhoc_dataset import WaveTrainDataset
from dataset import WaveEvalDataset, TrainDataLoader, EvalDataLoader
from adhoc_driver import Trainer
from models.wavesplit import SpeakerStack, SeparationStack
from adhoc_model import WaveSplit
from criterion.sdr import NegSDR, NegSISDR
from adhoc_criterion import SpeakerDistance, GlobalClassificationLoss, EntropyRegularizationLoss, MultiDomainLoss

parser = argparse.ArgumentParser(description="Training of WaveSplit")

parser.add_argument('--train_wav_root', type=str, default=None, help='Path for training dataset ROOT directory')
parser.add_argument('--valid_wav_root', type=str, default=None, help='Path for validation dataset ROOT directory')
parser.add_argument('--train_list_path', type=str, default=None, help='Path for mix_<n_sources>_spk_<max,min>_tr_mix')
parser.add_argument('--valid_list_path', type=str, default=None, help='Path for mix_<n_sources>_spk_<max,min>_cv_mix')
parser.add_argument('--sample_rate', '-sr', type=int, default=8000, help='Sampling rate')
parser.add_argument('--duration', type=float, default=0.75, help='Duration')
parser.add_argument('--valid_duration', type=float, default=4, help='Duration for valid dataset for avoiding memory error.')
parser.add_argument('--latent_dim', type=int, default=512, help='Latent dimension')
parser.add_argument('--spk_kernel_size', type=int, default=3, help='Kernel size of speaker stack.')
parser.add_argument('--spk_num_layers', type=int, default=14, help='# layers of speaker stack')
parser.add_argument('--sep_kernel_size_in', type=int, default=4, help='Kernel size')
parser.add_argument('--sep_kernel_size', type=int, default=3, help='Kernel size of separation stack.')
parser.add_argument('--sep_num_layers', type=int, default=10, help='# layers of separation stack')
parser.add_argument('--sep_num_blocks', type=int, default=4, help='# blocks of separation stack.')
parser.add_argument('--dilated', type=int, default=1, help='Dilated convolution')
parser.add_argument('--separable', type=int, default=1, help='Depthwise-separable convolution')
parser.add_argument('--causal', type=int, default=0, help='Causality')
parser.add_argument('--nonlinear', type=str, default=None, help='Non-linear function of separator')
parser.add_argument('--norm', type=int, default=1, help='Normalization')
parser.add_argument('--n_sources', type=int, default=None, help='# speakers')
parser.add_argument('--reconst_criterion', type=str, default='sdr', choices=['sdr','sisdr'], help='Criterion for reconstruction')
parser.add_argument('--spk_criterion', type=str, default='distance', choices=['distance', 'global'], help='Criterion for speaker loss')
parser.add_argument('--reg_criterion', type=str, default='entropy', choices=['entropy', 'none', None], help='Regularization for speaker embedding')
parser.add_argument('--optimizer', type=str, default='adam', choices=['sgd', 'adam', 'rmsprop'], help='Optimizer, [sgd, adam, rmsprop]')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate. Default: 1e-3')
parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay (L2 penalty). Default: 0')
parser.add_argument('--max_norm', type=float, default=None, help='Gradient clipping')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size. Default: 4')
parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
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
    overlap = samples // 2
    max_samples = int(args.sample_rate * args.valid_duration)
    
    spk_to_idx = create_spk_to_idx(args.train_list_path)
    train_dataset = WaveTrainDataset(args.train_wav_root, args.train_list_path, samples=samples, overlap=overlap, n_sources=args.n_sources, spk_to_idx=spk_to_idx)
    valid_dataset = WaveEvalDataset(args.valid_wav_root, args.valid_list_path, max_samples=max_samples, n_sources=args.n_sources)
    print("Training dataset includes {} samples. {} speakers.".format(len(train_dataset), len(train_dataset.spk_to_idx.table)))
    print("Valid dataset includes {} samples.".format(len(valid_dataset)))
    
    loader = {}
    loader['train'] = TrainDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    loader['valid'] = EvalDataLoader(valid_dataset, batch_size=1, shuffle=False)

    # Criterion
    if args.reconst_criterion == 'sdr':
        reconst_criterion = NegSDR()
    elif args.reconst_criterion == 'sisdr':
        reconst_criterion = NegSISDR()
    else:
        raise ValueError("Not support criterion {}".format(args.reconst_criterion))

    if args.spk_criterion == 'distance':
        spk_criterion = SpeakerDistance(n_sources=args.n_sources)
    elif args.spk_criterion == 'global':
        spk_criterion = GlobalClassificationLoss(n_sources=args.n_sources, source_reduction='mean')
    else:
        raise ValueError("Not support criterion {}".format(args.spk_criterion))

    if args.reg_criterion is None or args.reg_criterion == 'none':
        reg_criterion = None
    elif args.reg_criterion == 'entropy':
        reg_criterion = EntropyRegularizationLoss()
    else:
        raise ValueError("Not support criterion {}".format(args.reg_criterion))
    
    criterion = MultiDomainLoss(reconst_criterion, spk_criterion, reg_criterion=reg_criterion)
    
    args.in_channels = 1

    speaker_stack = SpeakerStack(
        args.in_channels, args.latent_dim,
        kernel_size=args.spk_kernel_size, num_layers=args.spk_num_layers,
        dilated=args.dilated, separable=args.separable, causal=args.causal, nonlinear=args.nonlinear, norm=args.norm,
        n_sources=args.n_sources
    )

    separation_stack = SeparationStack(
        args.in_channels, args.latent_dim,
        kernel_size_in=args.sep_kernel_size_in, kernel_size=args.sep_kernel_size, num_blocks=args.sep_num_blocks, num_layers=args.sep_num_layers,
        dilated=args.dilated, separable=args.separable, causal=args.causal, nonlinear=args.nonlinear, norm=args.norm,
        n_sources=args.n_sources
    )

    model = WaveSplit(
        speaker_stack, separation_stack,
        latent_dim=args.latent_dim,
        n_sources=args.n_sources, n_training_sources=len(train_dataset.spk_to_idx.table),
        spk_criterion=spk_criterion
    )
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

    if args.max_norm is not None and args.max_norm == 0:
        args.max_norm = None
    
    trainer = Trainer(model, loader, criterion, optimizer, args)
    trainer.run()
    
if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    main(args)
    