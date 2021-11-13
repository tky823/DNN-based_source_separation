import os
import time

import torch
import torchaudio
import torch.nn as nn

from utils.utils import draw_loss_curve
from criterion.pit import pit as pit_wrapper

BITS_PER_SAMPLE_WSJ0 = 16
HALVE_LR = 3
PATIENCE = 10

class Trainer:
    def __init__(self, model, loader, criterion, optimizer, args):
        self.train_loader, self.valid_loader = loader['train'], loader['valid']
        
        self.model = model
        
        self.criterion = criterion
        self.optimizer = optimizer
        
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
        
        self.return_all_layers = True
        self.use_cuda = args.use_cuda
        
        if args.continue_from:
            config = torch.load(args.continue_from, map_location=lambda storage, loc: storage)
            
            self.start_epoch = config['epoch']
            
            self.train_loss[:self.start_epoch] = config['train_loss'][:self.start_epoch]
            self.valid_loss[:self.start_epoch] = config['valid_loss'][:self.start_epoch]
            
            self.best_loss = config['best_loss']
            self.prev_loss = self.valid_loss[self.start_epoch-1]
            self.no_improvement = config['no_improvement']
            
            if isinstance(self.model, nn.DataParallel):
                self.model.module.load_state_dict(config['base']['state_dict'])
            else:
                self.model.load_state_dict(config['base']['state_dict'])
            
            self.optimizer.load_state_dict(config['optim_dict'])
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
            
            print("[Epoch {}/{}] loss (train): {:.5f}, loss (valid): {:.5f}, {:.3f} [sec]".format(epoch + 1, self.epochs, train_loss, valid_loss, end - start), flush=True)
            
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
                    if self.no_improvement >= PATIENCE:
                        print("Stop training")
                        break
                    if self.no_improvement >= HALVE_LR:
                        for param_group in self.optimizer.param_groups:
                            prev_lr = param_group['lr']
                            lr = 0.5 * prev_lr
                            print("Learning rate: {} -> {}".format(prev_lr, lr))
                            param_group['lr'] = lr
                else:
                    self.no_improvement = 0
            
            self.prev_loss = valid_loss
            
            model_path = os.path.join(self.model_dir, "last.pth")
            self.save_model(epoch, model_path)
            
            save_path = os.path.join(self.loss_dir, "loss.png")
            draw_loss_curve(train_loss=self.train_loss[:epoch + 1], valid_loss=self.valid_loss[:epoch + 1], save_path=save_path)
    
    def run_one_epoch(self, epoch):
        """
        Training
        """
        train_loss = self.run_one_epoch_train(epoch)
        valid_loss = self.run_one_epoch_eval(epoch)

        return train_loss, valid_loss
    
    def run_one_epoch_train(self, epoch):
        """
        Training
        """
        self.model.train()
        
        train_loss = 0
        n_train_batch = len(self.train_loader)
        
        for idx, (mixture, sources, spk_idx) in enumerate(self.train_loader):
            if self.use_cuda:
                mixture = mixture.cuda()
                sources = sources.cuda()
                spk_idx = spk_idx.cuda()
            
            self.optimizer.zero_grad()

            with torch.no_grad():
                sorted_idx = self.model(mixture, spk_idx=spk_idx)
            
            self.optimizer.zero_grad()
            estimated_sources, spk_vector, spk_embedding, all_spk_embedding = self.model(mixture, spk_idx=spk_idx, sorted_idx=sorted_idx, return_all_layers=self.return_all_layers, return_spk_vector=True, return_spk_embedding=True, return_all_spk_embedding=True)
            
            if self.return_all_layers:
                loss = self.criterion(estimated_sources, sources.unsqueeze(dim=1), spk_vector=spk_vector, spk_embedding=spk_embedding, all_spk_embedding=all_spk_embedding, batch_mean=True)
            else:
                loss = self.criterion(estimated_sources, sources, spk_vector=spk_vector, spk_embedding=spk_embedding, all_spk_embedding=all_spk_embedding, batch_mean=True)
            
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
        n_valid_batch = len(self.valid_loader)

        with torch.no_grad():
            for idx, (mixture, sources, segment_IDs) in enumerate(self.valid_loader):
                if self.use_cuda:
                    mixture = mixture.cuda()
                    sources = sources.cuda()
                
                estimated_sources = self.model(mixture, return_all_layers=False, return_spk_vector=False, return_spk_embedding=False, return_all_spk_embedding=False)
                
                loss, _ = pit_wrapper(self.criterion.reconst_criterion, estimated_sources, sources, batch_mean=False)
                loss = loss.sum(dim=0)
                valid_loss += loss.item()

                if idx < 5:
                    mixture = mixture[0].squeeze(dim=0).cpu()
                    estimated_sources = estimated_sources[0].cpu()
                    
                    save_dir = os.path.join(self.sample_dir, segment_IDs[0])
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
        
        valid_loss /= n_valid_batch
        
        return valid_loss
    
    def save_model(self, epoch, model_path='./tmp.pth'):
        config = {}
        
        if isinstance(self.model, nn.DataParallel):
            config['base'] = self.model.module.get_config()
            config['base']['state_dict'] = self.model.module.state_dict()

            config['speaker_stack'] = self.model.module.speaker_stack.get_config()
            config['separation_stack'] = self.model.module.separation_stack.get_config()

            config['speaker_stack']['state_dict'] = self.model.module.speaker_stack.state_dict()
            config['separation_stack']['state_dict'] = self.model.module.separation_stack.state_dict()
        else:
            config['base'] = self.model.get_config()
            config['base']['state_dict'] = self.model.state_dict()

            config['speaker_stack'] = self.model.speaker_stack.get_config()
            config['separation_stack'] = self.model.separation_stack.get_config()

            config['speaker_stack']['state_dict'] = self.model.speaker_stack.state_dict()
            config['separation_stack']['state_dict'] = self.model.separation_stack.state_dict()
            
        config['optim_dict'] = self.optimizer.state_dict()
        
        config['best_loss'] = self.best_loss
        config['no_improvement'] = self.no_improvement
        
        config['train_loss'] = self.train_loss
        config['valid_loss'] = self.valid_loss
        
        config['epoch'] = epoch + 1
        
        torch.save(config, model_path)

