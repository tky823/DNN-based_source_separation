import os

import musdb
import museval
import torch
import torchaudio
import torch.nn as nn

from driver import TrainerBase, TesterBase

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
            mean, std = mixture.mean(dim=-1, keepdim=True), mixture.std(dim=-1, keepdim=True)
            standardized_mixture = (mixture - mean) / (std + EPS)
            standardized_sources = (sources - mean) / (std + EPS)
            standardized_estimated_sources = self.model(standardized_mixture)
            loss = self.criterion(standardized_estimated_sources, standardized_sources)
            
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
                if self.use_cuda:
                    mixture = mixture.cuda()
                    sources = sources.cuda()
                mean, std = mixture.mean(dim=-1, keepdim=True), mixture.std(dim=-1, keepdim=True)
                standardized_mixture = (mixture - mean) / (std + EPS)
                standardized_sources = (sources - mean) / (std + EPS)
                standardized_estimated_sources = self.model(standardized_mixture)
                loss = self.criterion(standardized_estimated_sources, standardized_sources, batch_mean=False)
                loss = loss.sum(dim=0)
                valid_loss += loss.item()
                
                if idx < 5:
                    estimated_sources = std * standardized_estimated_sources + mean
                    
                    mixture = mixture[0].squeeze(dim=0).detach().cpu()
                    estimated_sources = estimated_sources[0].detach().cpu()
                    
                    save_dir = os.path.join(self.sample_dir, titles[0])
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, "mixture.wav")
                    torchaudio.save(save_path, mixture, sample_rate=self.sr, bits_per_sample=BITS_PER_SAMPLE_MUSDB18)
                    
                    save_dir = os.path.join(self.sample_dir, titles[0], "epoch{}".format(epoch + 1))
                    os.makedirs(save_dir, exist_ok=True)
                    for source_idx, estimated_source in enumerate(estimated_sources):
                        target = self.valid_loader.dataset.target[source_idx]
                        save_path = os.path.join(save_dir, "{}.wav".format(target))
                        torchaudio.save(save_path, estimated_source, sample_rate=self.sr, bits_per_sample=BITS_PER_SAMPLE_MUSDB18)
        
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
        config['sr'] = self.train_loader.dataset.sr
        
        torch.save(config, model_path)

class AdhocTester(TesterBase):
    def __init__(self, model, loader, criterion, args):
        super().__init__(model, loader, criterion, args)

    def _reset(self, args):
        self.sr = args.sr
        self.n_sources = args.n_sources
        self.sources = args.sources

        self.musdb18_root = args.musdb18_root
        
        self.estimates_dir = args.estimates_dir
        self.json_dir = args.json_dir
        
        if self.estimates_dir is not None:
            self.estimates_dir = os.path.abspath(args.estimates_dir)
            os.makedirs(self.estimates_dir, exist_ok=True)
        
        if self.json_dir is not None:
            self.json_dir = os.path.abspath(args.json_dir)
            os.makedirs(self.json_dir, exist_ok=True)
        
        self.use_estimate_all, self.use_evaluate_all = args.estimate_all, args.evaluate_all
        
        self.use_cuda = args.use_cuda
        
        config = torch.load(args.model_path, map_location=lambda storage, loc: storage)
        
        if isinstance(self.model, nn.DataParallel):
            self.model.module.load_state_dict(config['state_dict'])
        else:
            self.model.load_state_dict(config['state_dict'])
    
    def run(self):
        if self.use_estimate_all:
            self.estimate_all()
        if self.use_evaluate_all:
            self.evaluate_all()

    def estimate_all(self):
        self.model.eval()
        
        test_loss = 0
        test_loss_improvement = 0
        n_test = len(self.loader.dataset)

        s = "Title, Loss:"
        for target in self.sources:
            s += " ({})".format(target)
        
        s += ", loss improvement:"
        for target in self.sources:
            s += " ({})".format(target)
        
        print(s, flush=True)
        
        with torch.no_grad():
            for idx, (mixture, sources, name) in enumerate(self.loader):
                """
                    mixture: (1, 1, n_mics, T)
                    sources: (1, n_sources, n_mics, T)
                    name <str>: Artist and title of song
                """
                if self.use_cuda:
                    mixture = mixture.cuda()
                    sources = sources.cuda()
                
                estimated_sources = self.model(mixture) # (1, n_sources, n_mics, T)

                loss_mixture = self.criterion(mixture, sources, batch_mean=False) # (1, n_sources)
                loss = self.criterion(estimated_sources, sources, batch_mean=False) # (1, n_sources)
                loss_improvement = loss_mixture - loss # (n_sources,)

                mixture = mixture.squeeze(dim=0).cpu() # (n_sources, n_mics, T)
                estimated_sources = estimated_sources.squeeze(dim=0).cpu() # (n_sources, n_mics, T)

                track_dir = os.path.join(self.estimates_dir, name)
                os.makedirs(track_dir, exist_ok=True)

                for source_idx, target in enumerate(self.sources):
                    estimated_path = os.path.join(track_dir, "{}.wav".format(target))
                    estimated_source = estimated_sources[source_idx] # -> (n_mics, T)
                    signal = estimated_source.unsqueeze(dim=0) if estimated_source.dim() == 1 else estimated_source
                    torchaudio.save(estimated_path, signal, sample_rate=self.sr, bits_per_sample=BITS_PER_SAMPLE_MUSDB18)
                
                s = "{},".format(name)
                for idx, target in enumerate(self.sources):
                    s += " {:.3f}".format(loss[idx].item())
                
                s += ", loss improvement:"
                for idx, target in enumerate(self.sources):
                    s += " {:.3f}".format(loss_improvement[idx].item())

                print(s, flush=True)
                
                test_loss += loss # (n_sources,)
                test_loss_improvement += loss_improvement # (n_sources,)

        test_loss /= n_test
        test_loss_improvement /= n_test
        
        s = "Loss:"
        for idx, target in enumerate(self.sources):
            s += " ({}) {:.3f}".format(target, test_loss[idx].item())
        
        s += ", loss improvement:"
        for idx, target in enumerate(self.sources):
            s += " ({}) {:.3f}".format(target, test_loss_improvement[idx].item())

        print(s, flush=True)
    
    def evaluate_all(self):
        mus = musdb.DB(root=self.musdb18_root, subsets='test')
        
        results = museval.EvalStore(frames_agg='median', tracks_agg='median')

        for track in mus.tracks:
            name = track.name
            
            estimates = {}
            estimated_accompaniment = 0

            for target in self.sources:
                estimated_path = os.path.join(self.estimates_dir, name, "{}.wav".format(target))
                estimated, _ = torchaudio.load(estimated_path)
                estimated = estimated.numpy().transpose(1, 0)
                estimates[target] = estimated
                if target != 'vocals':
                    estimated_accompaniment += estimated

            estimates['accompaniment'] = estimated_accompaniment

            # Evaluate using museval
            scores = museval.eval_mus_track(track, estimates, output_dir=self.json_dir)
            results.add_track(scores)

            print(name)
            print(scores, flush=True)

        print(results)