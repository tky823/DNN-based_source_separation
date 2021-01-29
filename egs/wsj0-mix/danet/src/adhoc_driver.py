import os
import numpy as np
import torch
import torch.nn as nn

from driver import TrainerBase
from utils.utils_audio import write_wav
from algorithm.stft import BatchInvSTFT
from criterion.pit import pit

class AdhocTrainer(TrainerBase):
    def __init__(self, model, loader, criterion, optimizer, args):
        self.train_loader, self.valid_loader = loader['train'], loader['valid']
        
        self.model = model
        
        self.criterion = criterion
        self.optimizer = optimizer
        
        self._reset(args)
    
    def _reset(self, args):
        # Override
        super()._reset(args)
        
        self.n_bins = args.n_bins
        self.istft = BatchInvSTFT(args.fft_size, args.hop_size, window_fn=args.window_fn)
    
    def run_one_epoch_train(self, epoch):
        # Override
        """
        Training
        """
        self.model.train()
        
        train_loss = 0
        n_train_batch = len(self.train_loader)
        
        for idx, (mixture, sources, assignment, threshold_weight) in enumerate(self.train_loader):
            if self.use_cuda:
                mixture = mixture.cuda()
                sources = sources.cuda()
                assignment = assignment.cuda()
                threshold_weight = threshold_weight.cuda()
                
            real, imag = mixture[...,0], mixture[...,1]
            mixture_amplitude = torch.sqrt(real**2+imag**2)
            real, imag = sources[...,0], sources[...,1]
            sources_amplitude = torch.sqrt(real**2+imag**2)
            
            estimated_sources_amplitude = self.model(mixture_amplitude, assignment=assignment, threshold_weight=threshold_weight, n_sources=sources.size(1))
            print("estimated_sources_amplitude", torch.isnan(estimated_sources_amplitude).any(), flush=True)
            loss = self.criterion(estimated_sources_amplitude, sources_amplitude)
            print("loss", torch.isnan(loss).any(), flush=True)
            
            print("Before")
            for p in self.model.parameters():
                if torch.isnan(p).any():
                    raise ValueError("NaN")
            self.optimizer.zero_grad()
            loss.backward()

            print("After")
            for p in self.model.parameters():
                if torch.isnan(p).any():
                    raise ValueError("NaN")
            
            if self.max_norm:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
            
            self.optimizer.step()

            print("After2")
            for p in self.model.parameters():
                if torch.isnan(p).any():
                    raise ValueError("NaN")
            
            train_loss += loss.item()
            
            if (idx + 1)%100 == 0:
                print("[Epoch {}/{}] iter {}/{} loss: {:.5f}".format(epoch+1, self.epochs, idx+1, n_train_batch, loss.item()), flush=True)
        
        train_loss /= n_train_batch
        
        return train_loss
    
    def run_one_epoch_eval(self, epoch):
        # Override
        """
        Validation
        """
        n_sources = self.n_sources
        
        self.model.eval()
        
        valid_loss = 0
        n_valid = len(self.valid_loader.dataset)
        
        with torch.no_grad():
            for idx, (mixture, sources, assignment, threshold_weight) in enumerate(self.valid_loader):
                """
                mixture (batch_size, 1, 2*F_bin, T_bin)
                sources (batch_size, n_sources, 2*F_bin, T_bin)
                assignment (batch_size, n_sources, F_bin, T_bin)
                threshold_weight (batch_size, F_bin, T_bin)
                """
                if self.use_cuda:
                    mixture = mixture.cuda()
                    sources = sources.cuda()
                    threshold_weight = threshold_weight.cuda()
                    assignment = assignment.cuda()
                
                real, imag = mixture[...,0], mixture[...,1]
                mixture_amplitude = torch.sqrt(real**2+imag**2)
                real, imag = sources[...,0], sources[...,1]
                sources_amplitude = torch.sqrt(real**2+imag**2)
                
                output = self.model(mixture_amplitude, assignment=None, threshold_weight=threshold_weight, n_sources=n_sources)
                # At the test phase, assignment may be unknown.
                loss, _ = pit(output, sources_amplitude, batch_mean=False)
                loss = loss.sum(dim=0)
                valid_loss += loss.item()
                
                if idx < 5:
                    mixture = mixture[0].cpu() # -> (1, n_bins, n_frames, 2)
                    mixture_amplitude = mixture_amplitude[0].cpu() # -> (1, n_bins, n_frames)
                    estimated_sources_amplitude = output[0].cpu() # -> (n_sources, n_bins, n_frames)
                    ratio = estimated_sources_amplitude / mixture_amplitude
                    real, imag = mixture[...,0], mixture[...,1]
                    real, imag = ratio * real, ratio * imag
                    estimated_sources = torch.cat([real.unsqueeze(dim=3), imag.unsqueeze(dim=3)], dim=3) # -> (n_sources, n_bins, n_frames, 2)
                    estimated_sources = self.istft(estimated_sources) # -> (n_sources, T)
                    estimated_sources = estimated_sources.cpu().numpy()
                    
                    mixture = self.istft(mixture) # -> (1, T)
                    mixture = mixture.squeeze(dim=0).numpy() # -> (T,)
                    
                    save_dir = os.path.join(self.sample_dir, "{}".format(idx+1))
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, "mixture.wav")
                    norm = np.abs(mixture).max()
                    mixture = mixture / norm
                    write_wav(save_path, signal=mixture, sr=self.sr)
                    
                    for source_idx, estimated_source in enumerate(estimated_sources):
                        save_path = os.path.join(save_dir, "epoch{}-{}.wav".format(epoch+1,source_idx+1))
                        norm = np.abs(estimated_source).max()
                        estimated_source = estimated_source / norm
                        write_wav(save_path, signal=estimated_source, sr=self.sr)
        
        valid_loss /= n_valid
        
        return valid_loss