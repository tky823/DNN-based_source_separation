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
        self.stage = args.stage
    
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
            
            batch_size, n_sources, _, T = sources.size()

            mixture, sources = mixture.view(batch_size, T), sources.view(batch_size * n_sources, T)

            mixture_resampled, sources_resampled = [], []

            for idx in range(self.stage):
                _mixture, _sources = self.resamplers[idx](mixture), self.resamplers[idx](sources)
                _mixture, _sources = _mixture.view(batch_size, 1, -1), _sources.view(batch_size * n_sources, 1, -1)
                mixture_resampled.append(_mixture)
                sources_resampled.append(_sources)
            
            # Forward
            estimated_sources, latent_estimated = self.model.extract_latent(mixture_resampled, masking=True, max_stage=self.stage)
            reconstructed, _ = self.model.extract_latent(mixture_resampled, masking=False, max_stage=self.stage)
            _, latent_target = self.model.extract_latent(sources_resampled, masking=False, max_stage=self.stage)

            # Main loss
            # print("Main loss")
            main_loss = 0
            for _estimated_sources, _sources in zip(estimated_sources, sources_resampled):
                _sources = _sources.view(batch_size, n_sources, *_estimated_sources.size()[-2:])
                # print(_estimated_sources.size(), _sources.size())
                _loss = self.criterion.metrics['main'](_estimated_sources, _sources)
                # print(_loss.size())
                main_loss = main_loss + _loss

            # Reconstruction loss
            # print("Reconstruction loss")
            reconstruction_loss = 0
            for _reconstructed, _mixture in zip(reconstructed, mixture_resampled):
                _loss = self.criterion.metrics['reconstruction'](_reconstructed, _mixture)
                # print(_mixture.size(), _reconstructed.size())
                reconstruction_loss = reconstruction_loss + _loss
                # print(_loss.size())
            
            # Similarity loss
            # print("Similarity loss")
            similarity_loss = 0
            for _latent_estimated, _latent_target in zip(latent_estimated, latent_target):
                _latent_target = _latent_target.view(batch_size, n_sources, *_latent_target.size()[-2:])
                # print(_latent_estimated.size(), _latent_target.size())
                _loss = self.criterion.metrics['similarity'](_latent_estimated, _latent_target)
                # print(_loss.size())
                similarity_loss = similarity_loss + _loss

            # Dissimilarity loss
            # print("Dissimilarity loss")
            dissimilarity_loss = 0
            for _latent_estimated in latent_estimated:
                # print(_latent_estimated.size())
                _loss = self.criterion.metrics['dissimilarity'](_latent_estimated)
                # print(_loss.size())
                similarity_loss = similarity_loss + _loss
            
            loss = main_loss + self.criterion.weights['reconstruction'] * reconstruction_loss + self.criterion.weights['similarity'] * similarity_loss + self.criterion.weights['dissimilarity'] * dissimilarity_loss
            loss = loss.mean(dim=0)
            
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
