import os
import time

import torch
import torchaudio
import torch.nn as nn

from utils.utils import draw_loss_curve
from transforms.stft import istft
from driver import TrainerBase

SAMPLE_RATE_MUSDB18 = 44100
BITS_PER_SAMPLE = 16 # 16 bit
EPS = 1e-12

class AdhocTrainer(TrainerBase):
    def __init__(self, model, loader, criterion, optimizer, args):
        self.train_loader, self.valid_loader = loader['train'], loader['valid']
        
        self.model = model
        
        self.criterion = criterion
        self.optimizer = optimizer
        
        self._reset(args)
        
    def _reset(self, args):
        # Override
        self.sample_rate = args.sample_rate

        self.n_fft, self.hop_length = args.n_fft, args.hop_length    
        self.window = self.valid_loader.dataset.window
        self.normalize = self.valid_loader.dataset.normalize

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
        
        self.resampler = torchaudio.transforms.Resample(self.sample_rate, SAMPLE_RATE_MUSDB18)

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
    
    def run(self):
        for epoch in range(self.start_epoch, self.epochs):
            start = time.time()
            train_loss, valid_loss = self.run_one_epoch(epoch)
            end = time.time()
            
            print("[Epoch {}/{}] loss (train): {:.5f}, loss (valid): {:.5f}, {:.3f} [sec]".format(epoch + 1, self.epochs, train_loss, valid_loss, end - start), flush=True)
            
            self.train_loss[epoch] = train_loss
            self.valid_loss[epoch] = valid_loss
            
            if valid_loss < self.best_loss:
                self.best_loss = valid_loss
                self.no_improvement = 0
                model_path = os.path.join(self.model_dir, "best.pth")
                self.save_model(epoch, model_path)
            else:
                if valid_loss >= self.prev_loss:
                    self.no_improvement += 1
                    if self.no_improvement >= 10:
                        print("Stop training")
                        break
                else:
                    self.no_improvement = 0
            
            self.prev_loss = valid_loss
            
            model_path = os.path.join(self.model_dir, "last.pth")
            self.save_model(epoch, model_path)
            
            save_path = os.path.join(self.loss_dir, "loss.png")
            draw_loss_curve(train_loss=self.train_loss[:epoch + 1], valid_loss=self.valid_loss[:epoch + 1], save_path=save_path)
    
    def run_one_epoch_train(self, epoch):
        # Override
        """
        Training
        """
        self.model.train()
        
        train_loss = 0
        n_train_batch = len(self.train_loader)
        
        for idx, (mixture, target, latent) in enumerate(self.train_loader):
            if self.use_cuda:
                mixture = mixture.cuda()
                target = target.cuda()
                latent = latent.cuda()
            
            mixture_amplitude = torch.abs(mixture)
            target_amplitude = torch.abs(target)

            if self.model.masking:
                estimated_target_amplitude = self.model(mixture_amplitude, latent)
            else:
                estimated_mask = self.model(mixture_amplitude, latent)
                estimated_target_amplitude = estimated_mask * mixture_amplitude
            
            loss = self.criterion(estimated_target_amplitude, target_amplitude)
            
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
        # Override
        """
        Validation
        """
        self.model.eval()
        
        valid_loss = 0
        n_valid = len(self.valid_loader.dataset)

        with torch.no_grad():
            for idx, (mixture, target, latent, source_names, scales) in enumerate(self.valid_loader):
                """
                mixture (batch_size, n_mics, n_bins, n_frames)
                target (batch_size, n_mics, n_bins, n_frames)
                title <list<str>>
                """
                if self.use_cuda:
                    mixture = mixture.cuda()
                    target = target.cuda()
                    latent = latent.cuda()
                
                mixture_amplitude = torch.abs(mixture)
                target_amplitude = torch.abs(target)
                
                estimated_mask = self.model(mixture_amplitude, latent)
                estimated_target_amplitude = estimated_mask * mixture_amplitude
                loss = self.criterion(estimated_target_amplitude, target_amplitude, batch_mean=False)
                loss = loss.sum(dim=0)
                valid_loss += loss.item()
                
                if idx < 5:
                    save_dir = os.path.join(self.sample_dir, "{}".format(idx + 1))
                    os.makedirs(save_dir, exist_ok=True)

                    mixture = mixture[0].cpu() # (2, n_bins, n_frames)
                    phase = torch.angle(mixture)
                    mixture_amplitude = mixture_amplitude[0].cpu() # (2, n_bins, n_frames)
                    estimated_sources = estimated_target_amplitude * torch.exp(1j * phase) # (len(source_names), 2, n_bins, n_frames)

                    for idx, source_name in enumerate(source_names):
                        scale = scales[idx]
                        estimated_source = istft(estimated_sources[idx], self.n_fft, hop_length=self.hop_length, window=self.window, normalized=self.normalize, return_complex=False) # (2, T)
                        save_path = os.path.join(save_dir, "epoch{}_{}{}.wav".format(epoch + 1, source_name, scale))
                        estimated_source = self.resampler(estimated_source)
                        torchaudio.save(save_path, estimated_source, sample_rate=SAMPLE_RATE_MUSDB18, bits_per_sample=BITS_PER_SAMPLE)

                    mixture = istft(mixture, self.n_fft, hop_length=self.hop_length, window=self.window, normalized=self.normalize, return_complex=False) # (2, T)
                    save_path = os.path.join(save_dir, "mixture.wav")
                    mixture = self.resampler(mixture)
                    torchaudio.save(save_path, mixture, sample_rate=SAMPLE_RATE_MUSDB18, bits_per_sample=BITS_PER_SAMPLE)
        
        valid_loss /= n_valid
        
        return valid_loss