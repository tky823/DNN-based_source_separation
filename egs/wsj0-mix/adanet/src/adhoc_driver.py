import os
import time

import torch
import torchaudio
import torch.nn as nn

from utils.utils import draw_loss_curve
from transforms.stft import istft
from driver import TrainerBase
from criterion.pit import pit

BITS_PER_SAMPLE_WSJ0 = 16
MIN_PESQ = -0.5
NO_IMPROVEMENT = 10

class AdhocTrainer(TrainerBase):
    def __init__(self, model, loader, criterion, optimizer, scheduler, args):
        self.train_loader, self.valid_loader = loader['train'], loader['valid']
        
        self.model = model
        
        self.criterion = criterion
        self.optimizer, self.scheduler = optimizer, scheduler
        
        self._reset(args)

    def _reset(self, args):
        self.sample_rate = args.sample_rate
        self.n_sources = args.n_sources
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
            config = torch.load(args.continue_from, map_location=lambda storage, loc: storage)
            
            self.start_epoch = config['epoch']
            
            self.train_loss[:self.start_epoch] = config['train_loss'][:self.start_epoch]
            self.valid_loss[:self.start_epoch] = config['valid_loss'][:self.start_epoch]
            
            self.best_loss = config['best_loss']
            self.prev_loss = self.valid_loss[self.start_epoch - 1]
            self.no_improvement = config['no_improvement']
            
            if isinstance(self.model, nn.DataParallel):
                self.model.module.load_state_dict(config['state_dict'])
            else:
                self.model.load_state_dict(config['state_dict'])
            
            self.optimizer.load_state_dict(config['optim_dict'])
            self.scheduler.load_state_dict(config['scheduler_dict'])
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
        
        self.n_bins = args.n_bins
        self.n_fft, self.hop_length = args.n_fft, args.hop_length
        self.window = self.train_loader.dataset.window
        self.normalize = self.train_loader.dataset.normalize

        self.target_type = args.target_type
    
    def run(self):
        for epoch in range(self.start_epoch, self.epochs):
            start = time.time()
            train_loss, valid_loss = self.run_one_epoch(epoch)
            end = time.time()
            
            s = "[Epoch {}/{}] loss (train): {:.5f}".format(epoch + 1, self.epochs, train_loss)
            self.train_loss[epoch] = train_loss

            if self.valid_loader is not None:
                s += ", loss (valid): {:.5f}".format(valid_loss)
                self.valid_loss[epoch] = valid_loss
            
            s += ", {:.3f} [sec]".format(end - start)
            print(s, flush=True)

            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(valid_loss)
            else:
                self.scheduler.step()
            
            if self.valid_loader is not None:
                if valid_loss < self.best_loss:
                    self.best_loss = valid_loss
                    self.no_improvement = 0
                    model_path = os.path.join(self.model_dir, "best.pth")
                    self.save_model(epoch, model_path)
                else:
                    if valid_loss >= self.prev_loss:
                        self.no_improvement += 1
                        if self.no_improvement >= NO_IMPROVEMENT:
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
        n_sources = self.n_sources

        self.model.train()
        
        train_loss = 0
        n_train_batch = len(self.train_loader)
        
        for idx, (mixture, sources, ideal_mask, threshold_weight) in enumerate(self.train_loader):
            if self.use_cuda:
                mixture = mixture.cuda()
                ideal_mask = ideal_mask.cuda()
                threshold_weight = threshold_weight.cuda()
            
            mixture_amplitude = torch.abs(mixture)
            if self.target_type == "source":
                target_amplitude = torch.abs(sources)
            elif self.target_type == "oracle":
                target_amplitude = ideal_mask * mixture_amplitude
            else:
                raise NotImplementedError("Not support `target_type={}.`".format(self.target_type))
            
            estimated_sources_amplitude = self.model(mixture_amplitude, threshold_weight=threshold_weight, n_sources=n_sources)
            
            loss = self.criterion(estimated_sources_amplitude, target_amplitude)
            
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
            for idx, (mixture, sources, ideal_mask, threshold_weight) in enumerate(self.valid_loader):
                """
                    mixture (batch_size, 1, n_bins, n_frames)
                    sources (batch_size, n_sources, n_bins, n_frames)
                    ideal_mask (batch_size, n_sources, n_bins, n_frames)
                    threshold_weight (batch_size, n_bins, n_frames)
                """
                if self.use_cuda:
                    mixture = mixture.cuda()
                    sources = sources.cuda()
                    ideal_mask = ideal_mask.cuda()
                    threshold_weight = threshold_weight.cuda()
                
                mixture_amplitude = torch.abs(mixture)
                if self.target_type == "source":
                    target_amplitude = torch.abs(sources)
                elif self.target_type == "oracle":
                    target_amplitude = ideal_mask * mixture_amplitude
                else:
                    raise NotImplementedError("Not support `target_type={}.`".format(self.target_type))
                
                estimated_sources_amplitude = self.model(mixture_amplitude, threshold_weight=threshold_weight, n_sources=n_sources)
                # At the test phase, assignment may be unknown.
                loss, _ = pit(self.criterion, estimated_sources_amplitude, target_amplitude, batch_mean=False)
                loss = loss.sum(dim=0)
                valid_loss += loss.item()
                
                if idx < 5:
                    mixture = mixture[0].cpu() # (1, n_bins, n_frames, 2)
                    mixture_amplitude = mixture_amplitude[0].cpu() # (1, n_bins, n_frames)
                    estimated_sources_amplitude = estimated_sources_amplitude[0].cpu() # (n_sources, n_bins, n_frames)

                    phase = torch.angle(mixture)
                    estimated_sources = estimated_sources_amplitude * torch.exp(1j * phase)
                    estimated_sources = istft(estimated_sources, n_fft=self.n_fft, hop_length=self.hop_length, normalized=self.normalize, window=self.window) # (n_sources, T)
                    estimated_sources = estimated_sources.cpu()
                    
                    mixture = istft(mixture, n_fft=self.n_fft, hop_length=self.hop_length, normalized=self.normalize, window=self.window) # (1, T)
                    mixture = mixture.squeeze(dim=0) # (T,)
                    
                    save_dir = os.path.join(self.sample_dir, "{}".format(idx + 1))
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, "mixture.wav")
                    norm = torch.abs(mixture).max()
                    mixture = mixture / norm
                    signal = mixture.unsqueeze(dim=0) if mixture.dim() == 1 else mixture
                    torchaudio.save(save_path, signal, sample_rate=self.sample_rate, bits_per_sample=BITS_PER_SAMPLE_WSJ0)
                    
                    for source_idx, estimated_source in enumerate(estimated_sources):
                        save_path = os.path.join(save_dir, "epoch{}-{}.wav".format(epoch + 1, source_idx + 1))
                        norm = torch.abs(estimated_source).max()
                        estimated_source = estimated_source / norm
                        signal = estimated_source.unsqueeze(dim=0) if estimated_source.dim() == 1 else estimated_source
                        torchaudio.save(save_path, signal, sample_rate=self.sample_rate, bits_per_sample=BITS_PER_SAMPLE_WSJ0)
        
        valid_loss /= n_valid
        
        return valid_loss

    def save_model(self, epoch, model_path='./tmp.pth'):
        if isinstance(self.model, nn.DataParallel):
            config = self.model.module.get_config()
            config['state_dict'] = self.model.module.state_dict()
        else:
            config = self.model.get_config()
            config['state_dict'] = self.model.state_dict()
            
        config['optim_dict'] = self.optimizer.state_dict()
        config['scheduler_dict'] = self.scheduler.state_dict()
        
        config['best_loss'] = self.best_loss
        config['no_improvement'] = self.no_improvement
        
        config['train_loss'] = self.train_loss
        config['valid_loss'] = self.valid_loss
        
        config['epoch'] = epoch + 1
        
        torch.save(config, model_path)
