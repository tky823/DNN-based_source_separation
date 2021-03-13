import os
import time
import numpy as np
import torch
import torch.nn as nn

from utils.utils import draw_loss_curve
from utils.utils_audio import write_wav
from driver import TrainerBase
from algorithm.stft import BatchInvSTFT


class AdhocTrainer(TrainerBase):
    def __init__(self, model, loader, criterion, optimizer, args):
        self.train_loader, self.valid_loader = loader['train'], loader['valid']
        
        self.model = model
        
        self.criterion = criterion
        self.optimizer = optimizer
        
        self._reset(args)
        
    def _reset(self, args):
        # Override
        self.sr = args.sr
        self.n_bins = args.n_bins

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

        self.istft = BatchInvSTFT(args.fft_size, args.hop_size, window_fn=args.window_fn)
    
    def run(self):
        for epoch in range(self.start_epoch, self.epochs):
            start = time.time()
            train_loss, valid_loss = self.run_one_epoch(epoch)
            end = time.time()
            
            print("[Epoch {}/{}] loss (train): {:.5f}, loss (valid): {:.5f}, {:.3f} [sec]".format(epoch+1, self.epochs, train_loss, valid_loss, end - start), flush=True)
            
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
            draw_loss_curve(train_loss=self.train_loss[:epoch+1], valid_loss=self.valid_loss[:epoch+1], save_path=save_path)
    
    def run_one_epoch_train(self, epoch):
        # Override
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
                
            real, imag = mixture[...,0], mixture[...,1]
            mixture_amplitude = torch.sqrt(real**2 + imag**2)
            real, imag = sources[...,0], sources[...,1]
            sources_amplitude = torch.sqrt(real**2 + imag**2)
            
            estimated_sources_amplitude = self.model(mixture_amplitude)
            loss = self.criterion(estimated_sources_amplitude, sources_amplitude)
            
            self.optimizer.zero_grad()
            loss.backward()
            
            if self.max_norm:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
            
            self.optimizer.step()
            
            train_loss += loss.item()
            
            if (idx + 1) % 100 == 0:
                print("[Epoch {}/{}] iter {}/{} loss: {:.5f}".format(epoch+1, self.epochs, idx+1, n_train_batch, loss.item()), flush=True)
        
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
            for idx, (mixture, sources, T, title) in enumerate(self.valid_loader):
                """
                mixture (batch_size, n_mics, n_bins, n_frames)
                sources (batch_size, n_mics, n_bins, n_frames)
                title <list<str>>
                """
                if self.use_cuda:
                    mixture = mixture.cuda()
                    sources = sources.cuda()
                
                real, imag = mixture[...,0], mixture[...,1]
                mixture_amplitude = torch.sqrt(real**2 + imag**2)
                real, imag = sources[...,0], sources[...,1]
                sources_amplitude = torch.sqrt(real**2 + imag**2)
                
                output = self.model(mixture_amplitude)
                loss = self.criterion(output, sources_amplitude, batch_mean=False)
                loss = loss.sum(dim=0)
                valid_loss += loss.item()
                
                if idx < 5:
                    mixture = mixture[0].cpu() # -> (2, n_bins, n_frames, 2)
                    mixture_amplitude = mixture_amplitude[0].cpu() # -> (2, n_bins, n_frames)
                    estimated_sources_amplitude = output[0].cpu() # -> (2, n_bins, n_frames)
                    ratio = estimated_sources_amplitude / mixture_amplitude
                    real, imag = mixture[...,0], mixture[...,1]
                    real, imag = ratio * real, ratio * imag
                    
                    estimated_source = torch.cat([real.unsqueeze(dim=3), imag.unsqueeze(dim=3)], dim=3) # -> (2, n_bins, n_frames, 2)
                    estimated_source = self.istft(estimated_source) # -> (2, T)
                    estimated_source = estimated_source.cpu().numpy()
                    
                    mixture = self.istft(mixture) # -> (2, T)
                    mixture = mixture.cpu().numpy()
                    
                    save_dir = os.path.join(self.sample_dir, "{}".format(idx + 1))

                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, "mixture.wav")
                    norm = np.abs(mixture).max()
                    mixture = mixture / norm
                    write_wav(save_path, signal=mixture.T, sr=self.sr)

                    save_path = os.path.join(save_dir, "epoch{}.wav".format(epoch + 1))
                    norm = np.abs(estimated_source).max()
                    estimated_source = estimated_source / norm
                    write_wav(save_path, signal=estimated_source>T, sr=self.sr)
        
        valid_loss /= n_valid
        
        return valid_loss