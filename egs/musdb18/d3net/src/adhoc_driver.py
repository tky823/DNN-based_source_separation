import os
import time

import torch
import torchaudio
import torch.nn as nn

from utils.utils import draw_loss_curve
from driver import TrainerBase, TesterBase

BITS_PER_SAMPLE = 16

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

        self.fft_size, self.hop_size = args.fft_size, args.hop_size    
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
            
            mixture_amplitude = torch.abs(mixture)
            sources_amplitude = torch.abs(sources)
            
            estimated_sources_amplitude = self.model(mixture_amplitude)
            loss = self.criterion(estimated_sources_amplitude, sources_amplitude)
            
            self.optimizer.zero_grad()
            loss.backward()
            
            if self.max_norm:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
            
            self.optimizer.step()
            
            train_loss += loss.item()
            
            if (idx + 1) % 100 == 0:
                print("[Epoch {}/{}] iter {}/{} loss: {:.5f}".format(epoch+1, self.epochs, idx + 1, n_train_batch, loss.item()), flush=True)
        
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
            for idx, (mixture, source, name) in enumerate(self.valid_loader):
                """
                    mixture: (batch_size, n_mics, n_bins, n_frames)
                    sources: (batch_size, n_mics, n_bins, n_frames)
                    name <list<str>>: Artist and title of song
                """
                if self.use_cuda:
                    mixture = mixture.cuda()
                    source = source.cuda()
                
                mixture_amplitude = torch.abs(mixture)
                source_amplitude = torch.abs(source)
                
                estimated_source_amplitude = self.model(mixture_amplitude)
                loss = self.criterion(estimated_source_amplitude, source_amplitude, batch_mean=False)
                loss = loss.mean(dim=0)
                valid_loss += loss.item()

                if idx < 5:
                    ratio = estimated_source_amplitude / mixture_amplitude
                    estimated_source = ratio * mixture # -> (batch_size, n_mics, n_bins, n_frames)

                    mixture_channels = mixture.size()[:-2] # -> (batch_size, n_mics)
                    estimated_source_channels = estimated_source.size()[:-2] # -> (batch_size, n_mics)
                    mixture = mixture.view(-1, *mixture.size()[-2:]) # -> (batch_size * n_mics, n_bins, n_frames)
                    estimated_source = estimated_source.view(-1, *estimated_source.size()[-2:]) # -> (batch_size * n_mics, n_bins, n_frames)
                    
                    mixture = torch.istft(mixture, self.fft_size, hop_length=self.hop_size, window=self.window, normalized=self.normalize, return_complex=False) # -> (n_mics, T_segment)
                    estimated_source = torch.istft(estimated_source, self.fft_size, hop_length=self.hop_size, window=self.window, normalized=self.normalize, return_complex=False) # -> (n_mics, T_segment)

                    mixture = mixture.view(*mixture_channels, -1) # -> (batch_size, n_mics, T_segment)
                    estimated_source = estimated_source.view(*estimated_source_channels, -1) # -> (batch_size, n_mics, T_segment)
                    
                    batch_size, n_mics, T_segment = mixture.size()
                    
                    mixture = mixture.cpu()
                    mixture = mixture.permute(1, 0, 2) # -> (n_mics, batch_size, T_segment)
                    mixture = mixture.reshape(n_mics, batch_size * T_segment)

                    estimated_source = estimated_source.cpu()
                    estimated_source = estimated_source.permute(1, 0, 2) # -> (n_mics, batch_size, T_segment)
                    estimated_source = estimated_source.reshape(n_mics, batch_size * T_segment)
                    
                    save_dir = os.path.join(self.sample_dir, name)

                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, "mixture.wav")
                    norm = torch.abs(mixture).max()
                    mixture = mixture / norm
                    torchaudio.save(save_path, mixture, sample_rate=self.sr, bits_per_sample=BITS_PER_SAMPLE)
                    
                    save_path = os.path.join(save_dir, "epoch{}.wav".format(epoch + 1))
                    norm = torch.abs(estimated_source).max()
                    estimated_source = estimated_source / norm
                    torchaudio.save(save_path, estimated_source, sample_rate=self.sr, bits_per_sample=BITS_PER_SAMPLE)
        
        valid_loss /= n_valid
        
        return valid_loss

class AdhocTester(TesterBase):
    def __init__(self, model, loader, criterion, args):
        super().__init__(model, loader, criterion, args)

    def _reset(self, args):
        self.sr = args.sr

        self.fft_size, self.hop_size = args.fft_size, args.hop_size    
        self.window = self.loader.dataset.window
        self.normalize = self.loader.dataset.normalize
        
        self.out_dir = args.out_dir
        
        if self.out_dir is not None:
            self.out_dir = os.path.abspath(args.out_dir)
            os.makedirs(self.out_dir, exist_ok=True)
        
        self.use_cuda = args.use_cuda
        
        package = torch.load(args.model_path, map_location=lambda storage, loc: storage)
        
        if isinstance(self.model, nn.DataParallel):
            self.model.module.load_state_dict(package['state_dict'])
        else:
            self.model.load_state_dict(package['state_dict'])
    
    def run(self):
        self.model.eval()
        
        test_loss = 0
        test_loss_improvement = 0
        n_test = len(self.loader.dataset)
        
        with torch.no_grad():
            for idx, (mixture, source, T, name) in enumerate(self.loader):
                """
                    mixture: (batch_size, 2, n_bins, n_frames)
                    source: (batch_size, 2, n_bins, n_frames)
                    T <float>: Length in time domain
                    name <str>: Artist and title of song
                """
                if self.use_cuda:
                    mixture = mixture.cuda()
                    source = source.cuda()
                
                samples = int(self.sr * T)
                
                mixture_amplitude = torch.abs(mixture)
                source_amplitude = torch.abs(source)
                
                loss_mixture = self.criterion(mixture_amplitude, source_amplitude, batch_mean=False)
                loss_mixture = loss_mixture.mean(dim=0)
                
                estimated_source_amplitude = []

                # Serial operation
                for _mixture_amplitude in mixture_amplitude:
                    _mixture_amplitude = _mixture_amplitude.unsqueeze(dim=0)
                    _estimated_source_amplitude = self.model(_mixture_amplitude)
                    estimated_source_amplitude.append(_estimated_source_amplitude)
                
                estimated_source_amplitude = torch.cat(estimated_source_amplitude, dim=0)
                loss = self.criterion(estimated_source_amplitude, source_amplitude, batch_mean=False)
                loss = loss.mean(dim=0)

                loss_improvement = loss_mixture.item() - loss.item()

                ratio = estimated_source_amplitude / mixture_amplitude
                estimated_source = ratio * mixture # -> (batch_size, n_mics, n_bins, n_frames)

                estimated_source_channels = estimated_source.size()[:-2] # -> (batch_size, n_mics)
                estimated_source = estimated_source.view(-1, *estimated_source.size()[-2:]) # -> (batch_size * n_mics, n_bins, n_frames)
                
                estimated_source = torch.istft(estimated_source, self.fft_size, hop_length=self.hop_size, window=self.window, normalized=self.normalize, return_complex=False) # -> (n_mics, T)

                estimated_source = estimated_source.view(*estimated_source_channels, -1) # -> (batch_size, n_mics, T_segment)
                
                batch_size, n_mics, T_segment = estimated_source.size()
                
                estimated_source = estimated_source.cpu()
                estimated_source = estimated_source.permute(1, 0, 2) # -> (n_mics, batch_size, T_segment)
                estimated_source = estimated_source.reshape(n_mics, batch_size * T_segment)[:, :samples]
                
                # Estimated source
                target = self.loader.dataset.target
                save_dir = os.path.join(self.out_dir, name)
                os.makedirs(save_dir, exist_ok=True)
                estimated_path = os.path.join(save_dir, "{}.wav".format(target))
                torchaudio.save(estimated_path, estimated_source, sample_rate=self.sr, bits_per_sample=BITS_PER_SAMPLE)
                
                test_loss += loss.item()
                test_loss_improvement += loss_improvement

        test_loss /= n_test
        test_loss_improvement /= n_test
        
        print("Loss: {:.3f}, loss improvement: {:3f}".format(test_loss, test_loss_improvement))