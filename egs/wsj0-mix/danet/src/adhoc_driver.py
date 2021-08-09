import os
import time

import torch
import torchaudio
import torch.nn as nn

from utils.utils import draw_loss_curve
from driver import TrainerBase
from criterion.pit import pit

BITS_PER_SAMPLE_WSJ0 = 16

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

        self.fft_size, self.hop_size = args.fft_size, args.hop_size

        if args.window_fn:
            if args.window_fn == 'hann':
                self.window = torch.hann_window(self.fft_size, periodic=True)
            elif args.window_fn == 'hamming':
                self.window = torch.hamming_window(self.fft_size, periodic=True)
            else:
                raise ValueError("Invalid argument.")
        else:
            self.window = None
        
        self.normalize = self.train_loader.dataset.normalize
        assert self.normalize == self.valid_loader.dataset.normalize, "Nomalization of STFT is different between `train_loader.dataset` and `valid_loader.dataset`."

        self.lr_decay = (args.lr_end / args.lr)**(1/self.epochs)
    
    def run(self):
        for epoch in range(self.start_epoch, self.epochs):
            start = time.time()
            train_loss, valid_loss = self.run_one_epoch(epoch)
            end = time.time()
            
            print("[Epoch {}/{}] loss (train): {:.5f}, loss (valid): {:.5f}, {:.3f} [sec]".format(epoch + 1, self.epochs, train_loss, valid_loss, end - start), flush=True)
            
            self.train_loss[epoch] = train_loss
            self.valid_loss[epoch] = valid_loss

            # Learning rate scheduling
            # torch.optim.lr_scheduler.ExponentialLR may be useful.
            lr_decay = self.lr_decay
            for param_group in self.optimizer.param_groups:
                prev_lr = param_group['lr']
                lr = lr_decay * prev_lr
                print("Learning rate: {} -> {}".format(prev_lr, lr))
                param_group['lr'] = lr
            
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
        
        for idx, (mixture, sources, assignment, threshold_weight) in enumerate(self.train_loader):
            if self.use_cuda:
                mixture = mixture.cuda()
                sources = sources.cuda()
                assignment = assignment.cuda()
                threshold_weight = threshold_weight.cuda()
            
            mixture_amplitude = torch.abs(mixture)
            sources_amplitude = torch.abs(sources)
            
            estimated_sources_amplitude = self.model(mixture_amplitude, assignment=assignment, threshold_weight=threshold_weight, n_sources=sources.size(1))
            loss = self.criterion(estimated_sources_amplitude, sources_amplitude)
            
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
        n_sources = self.n_sources
        
        self.model.eval()
        
        valid_loss = 0
        n_valid = len(self.valid_loader.dataset)
        
        with torch.no_grad():
            for idx, (mixture, sources, assignment, threshold_weight) in enumerate(self.valid_loader):
                """
                mixture (batch_size, 1, n_bins, n_frames)
                sources (batch_size, n_sources, n_bins, n_frames)
                assignment (batch_size, n_sources, n_bins, n_frames)
                threshold_weight (batch_size, n_bins, n_frames)
                """
                if self.use_cuda:
                    mixture = mixture.cuda()
                    sources = sources.cuda()
                    threshold_weight = threshold_weight.cuda()
                    assignment = assignment.cuda()
                
                mixture_amplitude = torch.abs(mixture)
                sources_amplitude = torch.abs(sources)
                
                output = self.model(mixture_amplitude, assignment=None, threshold_weight=threshold_weight, n_sources=n_sources)
                # At the test phase, assignment may be unknown.
                loss, _ = pit(self.criterion, output, sources_amplitude, batch_mean=False)
                loss = loss.sum(dim=0)
                valid_loss += loss.item()
                
                if idx < 5:
                    mixture = mixture[0].cpu() # -> (1, n_bins, n_frames, 2)
                    mixture_amplitude = mixture_amplitude[0].cpu() # -> (1, n_bins, n_frames)
                    estimated_sources_amplitude = output[0].cpu() # -> (n_sources, n_bins, n_frames)
                    ratio = estimated_sources_amplitude / mixture_amplitude
                    estimated_sources = ratio * mixture
                    estimated_sources = torch.istft(estimated_sources, n_fft=self.fft_size, hop_length=self.hop_size, normalized=self.normalize, window=self.window) # -> (n_sources, T)
                    estimated_sources = estimated_sources.cpu()
                    
                    mixture = torch.istft(mixture, n_fft=self.fft_size, hop_length=self.hop_size, normalized=self.normalize, window=self.window) # -> (1, T)
                    mixture = mixture.squeeze(dim=0) # -> (T,)
                    
                    save_dir = os.path.join(self.sample_dir, "{}".format(idx+1))
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, "mixture.wav")
                    norm = torch.abs(mixture).max()
                    mixture = mixture / norm
                    signal = mixture.unsqueeze(dim=0) if mixture.dim() == 1 else mixture
                    torchaudio.save(save_path, signal, sample_rate=self.sr, bits_per_sample=BITS_PER_SAMPLE_WSJ0)
                    
                    for source_idx, estimated_source in enumerate(estimated_sources):
                        save_path = os.path.join(save_dir, "epoch{}-{}.wav".format(epoch + 1, source_idx + 1))
                        norm = torch.abs(estimated_source).max()
                        estimated_source = estimated_source / norm
                        signal = estimated_source.unsqueeze(dim=0) if estimated_source.dim() == 1 else estimated_source
                        torchaudio.save(save_path, signal, sample_rate=self.sr, bits_per_sample=BITS_PER_SAMPLE_WSJ0)
        
        valid_loss /= n_valid
        
        return valid_loss