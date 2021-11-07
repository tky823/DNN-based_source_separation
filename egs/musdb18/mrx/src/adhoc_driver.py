import os

import torch
import torchaudio
import torch.nn as nn

from driver import TrainerBase

BITS_PER_SAMPLE_MUSDB18 = 16
EPS = 1e-12

class AdhocTrainer(TrainerBase):
    def __init__(self, model, loader, criterion, optimizer, args):
        super().__init__(model, loader, criterion, optimizer, args)
    
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
            
            estimated_sources = self.model(mixture)
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
                """
                mixture: (batch_size, 1, n_mics, patch_samples)
                sources: (batch_size, len(sources), n_mics, patch_samples)
                """
                if self.use_cuda:
                    mixture = mixture.cuda()
                    sources = sources.cuda()

                estimated_sources = self.model(mixture)

                mixture = mixture.permute(1, 2, 0, 3) # (1, n_mics, batch_size * patch_samples)
                mixture = mixture.reshape(*mixture.size()[:-2], -1) # (1, n_mics, batch_size * patch_samples)

                sources = sources.permute(1, 2, 0, 3) # (n_sources, n_mics, batch_size * patch_samples)
                sources = sources.reshape(*sources.size()[:-2], -1) # (n_sources, n_mics, batch_size * patch_samples)

                estimated_sources = estimated_sources.permute(1, 2, 0, 3) # (n_sources, n_mics, batch_size * patch_samples)
                estimated_sources = estimated_sources.reshape(*estimated_sources.size()[:-2], -1) # (n_sources, n_mics, batch_size * patch_samples)

                loss = self.criterion(estimated_sources, sources, batch_mean=True)
                valid_loss += loss.item()
                
                if idx < 5:
                    mixture = mixture.squeeze(dim=0).detach().cpu()
                    estimated_sources = estimated_sources.detach().cpu()
                    
                    save_dir = os.path.join(self.sample_dir, titles)
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, "mixture.wav")

                    if self.save_normalized:
                        norm = torch.abs(mixture).max()
                        mixture = mixture / norm
                    
                    torchaudio.save(save_path, mixture, sample_rate=self.sample_rate, bits_per_sample=BITS_PER_SAMPLE_MUSDB18)
                    
                    save_dir = os.path.join(self.sample_dir, titles, "epoch{}".format(epoch + 1))
                    os.makedirs(save_dir, exist_ok=True)

                    for source_idx, estimated_source in enumerate(estimated_sources):
                        target = self.valid_loader.dataset.target[source_idx]
                        save_path = os.path.join(save_dir, "{}.wav".format(target))
                        
                        if self.save_normalized:
                            norm = torch.abs(estimated_source).max()
                            estimated_source = estimated_source / norm
                        
                        torchaudio.save(save_path, estimated_source, sample_rate=self.sample_rate, bits_per_sample=BITS_PER_SAMPLE_MUSDB18)
        
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
        
        config['best_loss'] = self.best_loss
        config['no_improvement'] = self.no_improvement
        
        config['train_loss'] = self.train_loss
        config['valid_loss'] = self.valid_loss
        
        config['epoch'] = epoch + 1
        config['sample_rate'] = self.train_loader.dataset.sample_rate
        config['sources'] = self.train_loader.dataset.sources
        
        torch.save(config, model_path)
