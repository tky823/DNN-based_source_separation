import os

import numpy as np
import torch
import torchaudio
import torch.nn as nn

from driver import TrainerBase, TesterBase

BITS_PER_SAMPLE_MUSDB18 = 16

class AdhocTrainer(TrainerBase):
    def __init__(self, model, loader, criterion, optimizer, args):
        super().__init__(model, loader, criterion, optimizer, args)

class Tester(TesterBase):
    def __init__(self, model, loader, criterion, args):
        super().__init__(model, loader, criterion, args)

class SingleTargetTrainer(TrainerBase):
    def __init__(self, model, loader, criterion, optimizer, args):
        self.train_loader, self.valid_loader = loader['train'], loader['valid']
        
        self.model = model
        
        self.criterion = criterion
        self.optimizer = optimizer
        
        self._reset(args)
    
    def _reset(self, args):
        self.sr = args.sr
        self.max_norm = args.max_norm
        
        self.model_dir = args.model_dir
        self.loss_dir = args.loss_dir
        self.sample_dir = args.sample_dir
        
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.loss_dir, exist_ok=True)
        os.makedirs(self.sample_dir, exist_ok=True)
        
        self.epochs = args.epochs
        
        self.train_loss = torch.empty(self.epochs)
        self.valid_loss = torch.empty(self.epochs)
        
        self.use_cuda = args.use_cuda
        
        if args.continue_from:
            package = torch.load(args.continue_from, map_location=lambda storage, loc: storage)
            
            self.start_epoch = package['epoch']
            
            self.train_loss[:self.start_epoch] = package['train_loss'][:self.start_epoch]
            self.valid_loss[:self.start_epoch] = package['valid_loss'][:self.start_epoch]
            
            self.best_loss = package['best_loss']
            self.prev_loss = self.valid_loss[self.start_epoch-1]
            self.no_improvement = package['no_improvement']
            
            if isinstance(self.model, nn.DataParallel):
                self.model.module.load_state_dict(package['state_dict'])
            else:
                self.model.load_state_dict(package['state_dict'])
            
            self.optimizer.load_state_dict(package['optim_dict'])
        else:
            model_path = os.path.join(self.model_dir, "best.pth")
            
            if os.path.exists(model_path):
                if args.overwrite:
                    print("Overwrite models.")
                else:
                    raise ValueError("{} already exists. If you continue to run, set --overwrite to be True.".format(model_path))
            
            self.start_epoch = 0
            
            self.best_loss = float('infinity')
            self.prev_loss = float('infinity')
            self.no_improvement = 0

    def run_one_epoch_train(self, epoch):
        """
        Training
        """
        self.model.train()
        
        train_loss = 0
        n_train_batch = len(self.train_loader)
        
        for idx, (mixture, sources) in enumerate(self.train_loader):
            if self.use_cuda:
                mixture = mixture.cuda()
                sources = sources.cuda()
            
            mixture = mixture.unsqueeze(dim=1)
            estimated_sources = self.model(mixture)
            estimated_sources = estimated_sources.squeeze(dim=1)
            loss = self.criterion(estimated_sources, sources)
            
            self.optimizer.zero_grad()
            loss.backward()
            
            if self.max_norm:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
            
            self.optimizer.step()
            
            train_loss += loss.item()
            
            if (idx + 1) % 100 == 0:
                print("[Epoch {}/{}] iter {}/{} loss: {:.5f}".format(epoch + 1, self.epochs, idx + 1, n_train_batch, loss.item()), flush=True)
        
        train_loss /= n_train_batch
        
        return train_loss

    def run_one_epoch_eval(self, epoch):
        """
        Validation
        """
        self.model.eval()
        
        valid_loss = 0
        n_valid = len(self.valid_loader.dataset)
        
        with torch.no_grad():
            for idx, (mixture, sources, titles) in enumerate(self.valid_loader):
                if self.use_cuda:
                    mixture = mixture.cuda()
                    sources = sources.cuda()
                output = self.model(mixture)
                loss = self.criterion(output, sources, batch_mean=False)
                loss = loss.sum(dim=0)
                valid_loss += loss.item()
                
                if idx < 5:
                    mixture = mixture[0].detach().cpu().numpy()
                    estimated_source = output[0].detach().cpu().numpy()
                    
                    save_dir = os.path.join(self.sample_dir, titles[0])

                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, "mixture.wav")
                    norm = np.abs(mixture).max()
                    mixture = mixture / norm
                    torchaudio.save(save_path, mixture, sample_rate=self.sr, bits_per_sample=BITS_PER_SAMPLE_MUSDB18)
                    
                    save_path = os.path.join(save_dir, "epoch{}.wav".format(epoch + 1))
                    norm = np.abs(estimated_source).max()
                    estimated_source = estimated_source / norm
                    torchaudio.save(save_path, estimated_source, sample_rate=self.sr, bits_per_sample=BITS_PER_SAMPLE_MUSDB18)
        
        valid_loss /= n_valid
        
        return valid_loss
    