import os

import torch
import torchaudio
import torch.nn as nn

from driver import TrainerBase

SAMPLE_RATE_MUSDB18 = 44100

class Trainer(TrainerBase):
    def __init__(self, model, loader, criterion, optimizer, args):
        super().__init__(model, loader, criterion, optimizer, args)
    
    def _reset(self, args):
        super()._reset(args)

        sr = args.sr

        resamplers = []

        for sr_target in sr:
            resamplers.append(torchaudio.transforms.Resample(SAMPLE_RATE_MUSDB18, sr_target))

        self.resamplers = nn.ModuleList(resamplers)
    
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
            
            batch_size, n_sources, T = sources.size()

            mixture, sources = mixture.view(batch_size, T), sources.view(batch_size * n_sources, T)

            mixture_resampled, sources_resampled = [], []

            for idx, _ in enumerate(self.sr):
                _mixture, _sources = self.resamplers[idx](mixture), self.resamplers[idx](sources)
                _mixture, _sources = _mixture.view(batch_size, 1, -1), _sources.view(batch_size * n_sources, 1, -1)
                mixture_resampled.append(_mixture)
                sources_resampled.append(_sources)
            
            estimated_sources, latent_estimated = self.model.extract_latent(mixture_resampled, masking=True)
            reconstructed, _ = self.model.extract_latent(mixture_resampled, masking=False)
            _, latent_target = self.model.extract_latent(sources_resampled, masking=False)

            sdr_loss = 0

            for _estimated_sources, _sources in zip(estimated_sources, sources_resampled):
                _sources = _sources.view(batch_size, n_sources, *_estimated_sources.size()[-2:])
                _sdr_loss = self.criterion.metrics['neg_sisdr'](_estimated_sources, _sources).sum()
                sdr_loss = sdr_loss + _sdr_loss
                print(_estimated_sources.size(), _sources.size())    

            latent_loss = 0

            for _estimated_sources, _latent_estimated, _latent_target in zip(estimated_sources, latent_estimated, latent_target):
                _latent_target = _latent_target.view(batch_size, n_sources, *_latent_target.size()[-2:])
                _latent_loss = self.criterion.metrics['mse'](_latent_estimated, _latent_target).sum()
                latent_loss = latent_loss + _latent_loss
                print(_estimated_sources.size(), _latent_estimated.size(), _latent_target.size(), _latent_loss.size())

            reconstruceted_loss = 0
            for _mixture, _reconstructed in zip(mixture_resampled, reconstructed):
                _reconstructed_loss = self.criterion.metrics['mse'](_reconstructed, _mixture).sum()
                reconstruceted_loss = reconstruceted_loss + _reconstructed_loss
                print(_mixture.size(), _reconstructed.size(), _reconstructed_loss.size())
            
            loss = self.criterion.weights['neg_sisdr'] * sdr_loss + self.criterion.weights['mse'] * latent_loss + self.criterion.weights['mse'] * reconstruceted_loss
            
            exit()
            
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
        return 0
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
                    mixture = mixture[0].squeeze(dim=0).detach().cpu()
                    estimated_sources = output[0].detach().cpu()
                    
                    save_dir = os.path.join(self.sample_dir, titles[0])
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, "mixture.wav")
                    torchaudio.save(save_path, mixture, sample_rate=self.sr, bits_per_sample=BITS_PER_SAMPLE_MUSDB18)
                    
                    for source_idx, estimated_source in enumerate(estimated_sources):
                        target = self.valid_loader.dataset.target[source_idx]
                        save_path = os.path.join(save_dir, "epoch{}-{}.wav".format(epoch + 1, target))
                        torchaudio.save(save_path, estimated_source, sample_rate=self.sr, bits_per_sample=BITS_PER_SAMPLE_MUSDB18)
        
        valid_loss /= n_valid
        
        return valid_loss
