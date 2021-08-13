import os
import time

import torch
import torchaudio
import torch.nn as nn

from utils.utils import draw_loss_curve
from driver import TrainerBase

SAMPLE_RATE_MUSDB18 = 44100
BITS_PER_SAMPLE_MUSDB18 = 16

class Trainer(TrainerBase):
    def __init__(self, model, loader, criterion, optimizer, args):
        super().__init__(model, loader, criterion, optimizer, args)

    def _reset(self, args):
        self.sr = args.sr

        resamplers = []

        for sr_target in self.sr:
            resamplers.append(torchaudio.transforms.Resample(SAMPLE_RATE_MUSDB18, sr_target))

        self.resamplers = nn.ModuleList(resamplers)
        self.stage = args.stage

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
            package = torch.load(args.continue_from, map_location=lambda storage, loc: storage)
            
            self.start_epoch = package['epoch']
            
            for key in ['loss', 'main', 'reconstruction', 'similarity', 'dissimilarity']:
                self.train_loss[key][:self.start_epoch] = package['train_loss'][key][:self.start_epoch]
                self.valid_loss[key][:self.start_epoch] = package['valid_loss'][key][:self.start_epoch]
            
            self.best_loss = package['best_loss']
            self.prev_loss = self.valid_loss['loss'][self.start_epoch-1]
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
            
            print("[Epoch {}/{}] loss (train): {:.5f}, loss (valid): {:.5f}, {:.3f} [sec]".format(epoch + 1, self.epochs, train_loss['loss'], valid_loss['loss'], end - start), flush=True)
            
            for key in ['loss', 'main', 'reconstruction', 'similarity', 'dissimilarity']:
                self.train_loss[key][epoch] = train_loss[key]
                self.valid_loss[key][epoch] = valid_loss[key]
            
            if valid_loss['loss'] < self.best_loss:
                self.best_loss = valid_loss['loss']
                self.no_improvement = 0
                model_path = os.path.join(self.model_dir, "best.pth")
                self.save_model(epoch, model_path)
            else:
                if valid_loss >= self.prev_loss:
                    self.no_improvement += 1
                    if self.no_improvement >= 10:
                        print("Stop training")
                        break
                    if self.no_improvement >= 3:
                        for param_group in self.optimizer.param_groups:
                            prev_lr = param_group['lr']
                            lr = 0.5 * prev_lr
                            print("Learning rate: {} -> {}".format(prev_lr, lr))
                            param_group['lr'] = lr
                else:
                    self.no_improvement = 0
            
            self.prev_loss = valid_loss['loss']
            
            model_path = os.path.join(self.model_dir, "last.pth")
            self.save_model(epoch, model_path)

            for key in ['loss', 'main', 'reconstruction', 'similarity', 'dissimilarity']:
                save_path = os.path.join(self.loss_dir, "{}.png".format(key))
                draw_loss_curve(train_loss=self.train_loss[key][:epoch+1], valid_loss=self.valid_loss[key][:epoch+1], save_path=save_path)

    def run_one_epoch(self, epoch):
        """
        Training
        """
        train_loss, train_main_loss, train_reconstruction_loss, train_similarity_loss, train_similarity_loss, train_dissimilarity_loss = self.run_one_epoch_train(epoch)
        valid_loss, valid_main_loss, valid_reconstruction_loss, valid_similarity_loss, valid_similarity_loss, valid_dissimilarity_loss = self.run_one_epoch_eval(epoch)
        
        train_loss = {
            'loss': train_loss,
            'main': train_main_loss,
            'reconstruction': train_reconstruction_loss,
            'similarity': train_similarity_loss,
            'dissimilarity': train_dissimilarity_loss
        }
        valid_loss = {
            'loss': valid_loss,
            'main': valid_main_loss,
            'reconstruction': valid_reconstruction_loss,
            'similarity': valid_similarity_loss,
            'dissimilarity': valid_dissimilarity_loss
        }

        return train_loss, valid_loss
    
    def run_one_epoch_train(self, epoch):
        """
        Training
        """
        self.model.train()
        
        train_loss = 0
        train_main_loss = 0
        train_reconstruction_loss = 0
        train_similarity_loss = 0
        train_dissimilarity_loss = 0

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
            main_loss = 0
            for _estimated_sources, _sources in zip(estimated_sources, sources_resampled):
                _sources = _sources.view(batch_size, n_sources, *_estimated_sources.size()[-2:])
                _loss = self.criterion.metrics['main'](_estimated_sources, _sources)
                main_loss = main_loss + _loss

            # Reconstruction loss
            reconstruction_loss = 0
            for _reconstructed, _mixture in zip(reconstructed, mixture_resampled):
                _loss = self.criterion.metrics['reconstruction'](_reconstructed, _mixture)
                reconstruction_loss = reconstruction_loss + _loss
            
            # Similarity and dissimilarity loss
            similarity_loss, dissimilarity_loss = 0, 0
            for _latent_estimated, _latent_target in zip(latent_estimated, latent_target):
                _latent_target = _latent_target.view(batch_size, n_sources, *_latent_target.size()[-2:])

                _loss = self.criterion.metrics['similarity'](_latent_estimated, _latent_target)
                similarity_loss = similarity_loss + _loss

                _loss = self.criterion.metrics['dissimilarity'](_latent_estimated)
                dissimilarity_loss = dissimilarity_loss + _loss
            
            loss = main_loss + self.criterion.weights['reconstruction'] * reconstruction_loss + self.criterion.weights['similarity'] * similarity_loss + self.criterion.weights['dissimilarity'] * dissimilarity_loss
            loss = loss.mean(dim=0)
            
            self.optimizer.zero_grad()
            loss.backward()
            
            if self.max_norm:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
            
            self.optimizer.step()
            
            train_loss += loss.item()
            train_main_loss += main_loss.item()
            train_reconstruction_loss += reconstruction_loss.item()
            train_similarity_loss += similarity_loss.item()
            train_dissimilarity_loss += dissimilarity_loss.item()
            
            if (idx + 1) % 100 == 0:
                print("[Epoch {}/{}] iter {}/{} loss: {:.5f} (main: {:.5f}, reconstruction: {:.5f}, similarity: {:.5f}, dissimilarity: {:.5f})".format(epoch + 1, self.epochs, idx + 1, n_train_batch, loss.item(), main_loss.item(), reconstruction_loss.item(), similarity_loss.item(), dissimilarity_loss.item()), flush=True)
        
        train_loss /= n_train_batch
        train_main_loss /= n_train_batch
        train_reconstruction_loss /= n_train_batch
        train_similarity_loss /= n_train_batch
        train_dissimilarity_loss /= n_train_batch
        
        return train_loss, train_main_loss, train_reconstruction_loss, train_similarity_loss, train_similarity_loss, train_dissimilarity_loss
    
    def run_one_epoch_eval(self, epoch):
        """
        Validation
        """
        self.model.eval()
        
        valid_loss = 0
        valid_main_loss = 0
        valid_reconstruction_loss = 0
        valid_similarity_loss = 0
        valid_dissimilarity_loss = 0
        
        n_valid = len(self.valid_loader.dataset)
        
        with torch.no_grad():
            for idx, (mixture, sources, titles) in enumerate(self.valid_loader):
                if self.use_cuda:
                    mixture = mixture.cuda()
                    sources = sources.cuda()

                print(mixture.size(), sources.size())
                batch_size, n_sources, T = sources.size()
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
                main_loss = 0
                for _estimated_sources, _sources in zip(estimated_sources, sources_resampled):
                    _sources = _sources.view(batch_size, n_sources, *_estimated_sources.size()[-2:])
                    _loss = self.criterion.metrics['main'](_estimated_sources, _sources, batch_mean=False)
                    main_loss = main_loss + _loss

                # Reconstruction loss
                reconstruction_loss = 0
                for _reconstructed, _mixture in zip(reconstructed, mixture_resampled):
                    _loss = self.criterion.metrics['reconstruction'](_reconstructed, _mixture, batch_mean=False)
                    reconstruction_loss = reconstruction_loss + _loss
                
                # Similarity and dissimilarity loss
                similarity_loss, dissimilarity_loss = 0, 0
                for _latent_estimated, _latent_target in zip(latent_estimated, latent_target):
                    _latent_target = _latent_target.view(batch_size, n_sources, *_latent_target.size()[-2:])

                    _loss = self.criterion.metrics['similarity'](_latent_estimated, _latent_target, batch_mean=False)
                    similarity_loss = similarity_loss + _loss

                    _loss = self.criterion.metrics['dissimilarity'](_latent_estimated, batch_mean=False)
                    dissimilarity_loss = dissimilarity_loss + _loss
            
                loss = main_loss + self.criterion.weights['reconstruction'] * reconstruction_loss + self.criterion.weights['similarity'] * similarity_loss + self.criterion.weights['dissimilarity'] * dissimilarity_loss
                loss = loss.sum(dim=0)
                valid_loss += loss.item()

                valid_loss += loss.item()
                valid_main_loss += main_loss.item()
                valid_reconstruction_loss += reconstruction_loss.item()
                valid_similarity_loss += similarity_loss.item()
                valid_dissimilarity_loss += dissimilarity_loss.item()
                
                if idx < 5:
                    for stage_idx in range(self.stage):
                        _mixture_resampled, _estimated_sources = mixture_resampled[stage_idx], estimated_sources[stage_idx]
                        _sr = self.sr[stage_idx]

                        print(_mixture_resampled.size(), _estimated_sources.size(), _sr)
                        raise NotImplementedError
                        _mixture_resampled = _mixture_resampled[0].squeeze(dim=0).cpu()
                        _estimated_sources = _estimated_sources[0].cpu()
                        
                        save_dir = os.path.join(self.sample_dir, titles[0])
                        os.makedirs(save_dir, exist_ok=True)
                        save_path = os.path.join(save_dir, "mixture.wav")
                        torchaudio.save(save_path, _mixture_resampled, sample_rate=self.sr, bits_per_sample=BITS_PER_SAMPLE_MUSDB18)
                        
                        for source_idx, _estimated_source in enumerate(_estimated_sources):
                            target = self.valid_loader.dataset.target[source_idx]
                            save_path = os.path.join(save_dir, "epoch{}-{}.wav".format(epoch + 1, target))
                            torchaudio.save(save_path, _estimated_source, sample_rate=self.sr, bits_per_sample=BITS_PER_SAMPLE_MUSDB18)
            
        valid_loss /= n_valid
        valid_main_loss /= n_valid
        valid_reconstruction_loss /= n_valid
        valid_similarity_loss /= n_valid
        valid_dissimilarity_loss /= n_valid
        
        return valid_loss, valid_main_loss, valid_reconstruction_loss, valid_similarity_loss, valid_dissimilarity_loss