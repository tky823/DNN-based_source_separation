import os
import subprocess
import time
import numpy as np
import torch
import torch.nn as nn

from utils.utils import draw_loss_curve
from utils.utils_audio import write_wav
from algorithm.stft import BatchInvSTFT

class Trainer:
    def __init__(self, model, loader, pit_criterion, optimizer, args):
        self.train_loader, self.valid_loader = loader['train'], loader['valid']
        
        self.model = model
        
        self.pit_criterion = pit_criterion
        self.optimizer = optimizer
        
        self._reset(args)
    
    def _reset(self, args):
        self.sr = args.sr
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
            
            self.best_loss = package['best_loss']
            self.no_improvement = package['no_improvement']
            
            self.start_epoch = package['epoch']
            
            self.train_loss[:self.start_epoch] = package['train_loss'][:self.start_epoch]
            self.valid_loss[:self.start_epoch] = package['valid_loss'][:self.start_epoch]
            
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
            
            self.best_loss = float('infinity')
            self.no_improvement = 0
            
            self.start_epoch = 0
    
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
                self.no_improvement += 1
                if self.no_improvement >= 5:
                    print("Stop training")
                    break
                if self.no_improvement == 3:
                    optim_dict = self.optimizer.state_dict()
                    lr = optim_dict['param_groups'][0]['lr']
                    
                    print("Learning rate: {} -> {}".format(lr, 0.5 * lr))
                    
                    optim_dict['param_groups'][0]['lr'] = 0.5 * lr
                    self.optimizer.load_state_dict(optim_dict)
            
            model_path = os.path.join(self.model_dir, "last.pth")
            self.save_model(epoch, model_path)
            
            save_path = os.path.join(self.loss_dir, "loss.png")
            draw_loss_curve(train_loss=self.train_loss[:epoch+1], valid_loss=self.valid_loss[:epoch+1], save_path=save_path)
    
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
        
        for idx, (mixture, sources) in enumerate(self.train_loader):
            if self.use_cuda:
                mixture = mixture.cuda()
                sources = sources.cuda()
            
            estimated_sources = self.model(mixture)
            loss, _ = self.pit_criterion(estimated_sources, sources)
            
            self.optimizer.zero_grad()
            loss.backward()
            
            if self.max_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
            self.optimizer.step()
            
            train_loss += loss.item()
            
            if (idx + 1)%100 == 0:
                print("[Epoch {}/{}] iter {}/{} loss: {:.5f}".format(epoch+1, self.epochs, idx+1, n_train_batch, loss.item()), flush=True)
        
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
            for idx, (mixture, sources) in enumerate(self.valid_loader):
                if self.use_cuda:
                    mixture = mixture.cuda()
                    sources = sources.cuda()
                output = self.model(mixture)
                loss, _ = self.pit_criterion(output, sources, batch_mean=False)
                loss = loss.sum(dim=0)
                valid_loss += loss.item()
                
                if idx < 5:
                    estimated_sources = output[0].detach().cpu().numpy()
                    for source_idx, estimated_source in enumerate(estimated_sources):
                        save_dir = os.path.join(self.sample_dir, "{}".format(idx+1))
                        os.makedirs(save_dir, exist_ok=True)
                        save_path = os.path.join(save_dir, "epoch{}-{}.wav".format(epoch+1,source_idx))
                        norm = np.abs(estimated_source).max()
                        estimated_source = estimated_source / norm
                        write_wav(save_path, signal=estimated_source, sr=self.sr)
        
        valid_loss /= n_valid
        
        return valid_loss
    
    def save_model(self, epoch, model_path='./tmp.pth'):
        if isinstance(self.model, nn.DataParallel):
            package = self.model.module.get_package()
            package['state_dict'] = self.model.module.state_dict()
        else:
            package = self.model.get_package()
            package['state_dict'] = self.model.state_dict()
            
        package['optim_dict'] = self.optimizer.state_dict()
        
        package['best_loss'] = self.best_loss
        package['no_improvement'] = self.no_improvement
        
        package['train_loss'] = self.train_loss
        package['valid_loss'] = self.valid_loss
        
        package['epoch'] = epoch + 1
        
        torch.save(package, model_path)

class Tester:
    def __init__(self, model, loader, pit_criterion, args):
        self.loader = loader
        
        self.model = model
        
        self.pit_criterion = pit_criterion
        
        self._reset(args)
        
    def _reset(self, args):
        self.sr = args.sr
        self.n_sources = args.n_sources
        
        self.out_dir = args.out_dir
        
        if self.out_dir is not None:
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
        test_pesq = 0
        n_test = len(self.loader.dataset)
        
        with torch.no_grad():
            for idx, (mixture, sources, segment_IDs) in enumerate(self.loader):
                if self.use_cuda:
                    mixture = mixture.cuda()
                    sources = sources.cuda()
                
                loss_mixture, _ = self.pit_criterion(mixture, sources, batch_mean=False)
                loss_mixture = loss_mixture.sum(dim=0)
                
                output = self.model(mixture)
                loss, perm_idx = self.pit_criterion(output, sources, batch_mean=False)
                loss = loss.sum(dim=0)
                test_loss += loss.item()
                test_loss_improvement += loss_mixture.item() - loss.item()
                
                mixture = mixture[0].squeeze(dim=0).cpu().numpy() # -> (T,)
                sources = sources[0].cpu().numpy() # -> (n_sources, T)
                estimated_sources = output[0].cpu().numpy() # -> (n_sources, T)
                perm_idx = perm_idx[0] # -> (n_sources,)
                segment_IDs = segment_IDs[0] # -> (n_sources,)
                
                norm = np.abs(mixture).max()
                mixture /= norm
                mixture_ID = "+".join(segment_IDs)
                
                if idx < 10 and self.out_dir is not None:
                    mixture_path = os.path.join(self.out_dir, "{}.wav".format(mixture_ID))
                    write_wav(mixture_path, signal=mixture, sr=self.sr)
                mixture_path = "tmp-mixture.wav"
                write_wav(mixture_path, signal=mixture, sr=self.sr)
                
                for order_idx in range(self.n_sources):
                    source, estimated_source = sources[order_idx], estimated_sources[perm_idx[order_idx]]
                    segment_ID = segment_IDs[order_idx]
                    
                    # Target
                    norm = np.abs(source).max()
                    source /= norm
                    if idx < 10 and  self.out_dir is not None:
                        source_path = os.path.join(self.out_dir, "{}_{}-target.wav".format(mixture_ID, order_idx))
                        write_wav(source_path, signal=source, sr=self.sr)
                    source_path = "tmp-{}-target.wav".format(order_idx)
                    write_wav(source_path, signal=source, sr=self.sr)
                    
                    # Estimated source
                    norm = np.abs(estimated_source).max()
                    estimated_source /= norm
                    if idx < 10 and  self.out_dir is not None:
                        estimated_path = os.path.join(self.out_dir, "{}_{}-estimated.wav".format(mixture_ID, order_idx))
                        write_wav(estimated_path, signal=estimated_source, sr=self.sr)
                    estimated_path = "tmp-{}-estimated.wav".format(order_idx)
                    write_wav(estimated_path, signal=estimated_source, sr=self.sr)
                
                pesq = 0
                
                for source_idx in range(self.n_sources):
                    source_path = "tmp-{}-target.wav".format(source_idx)
                    estimated_path = "tmp-{}-estimated.wav".format(source_idx)
                    
                    command = "./PESQ +{} {} {}".format(self.sr, source_path, estimated_path)
                    command += " | grep Prediction | awk '{print $5}'"
                    pesq_output = subprocess.check_output(command, shell=True)
                    pesq_output = pesq_output.decode().strip()
                    pesq += float(pesq_output)
                    
                    subprocess.call("rm {}".format(source_path), shell=True)
                    subprocess.call("rm {}".format(estimated_path), shell=True)
                
                pesq /= self.n_sources
                print("{}, {:.3f}".format(mixture_ID, pesq), flush=True)
                
                test_pesq += pesq
        
        test_loss /= n_test
        test_loss_improvement /= n_test
        test_pesq /= n_test
            
        print("Loss: {:.3f}, loss improvement: {:3f} PESQ: {:.3f}".format(test_loss, test_loss_improvement, test_pesq))

class AttractorTrainer(Trainer):
    def __init__(self, model, loader, pit_criterion, optimizer, args):
        super().__init__(model, loader, pit_criterion, optimizer, args)
        
        self.F_bin = args.F_bin
        self.istft = BatchInvSTFT(args.fft_size, args.hop_size, window_fn=args.window_fn)
    
    def run_one_epoch_train(self, epoch):
        # Override
        """
        Training
        """
        F_bin = self.F_bin
        
        self.model.train()
        
        train_loss = 0
        n_train_batch = len(self.train_loader)
        
        for idx, (mixture, sources, assignment, threshold_weight) in enumerate(self.train_loader):
            if self.use_cuda:
                mixture = mixture.cuda()
                sources = sources.cuda()
                assignment = assignment.cuda()
                threshold_weight = threshold_weight.cuda()
            
            real, imag = mixture[:,:,:F_bin], mixture[:,:,F_bin:]
            mixture = torch.sqrt(real**2+imag**2)
            real, imag = sources[:,:,:F_bin], sources[:,:,F_bin:]
            sources = torch.sqrt(real**2+imag**2)
            
            estimated_sources = self.model(mixture, assignment=assignment, threshold_weight=threshold_weight)
            loss, _ = self.pit_criterion(estimated_sources, sources)
            
            self.optimizer.zero_grad()
            loss.backward()
            
            if self.max_norm:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
            self.optimizer.step()
            
            train_loss += loss.item()
            
            if (idx + 1)%100 == 0:
                print("[Epoch {}/{}] iter {}/{} loss: {:.5f}".format(epoch+1, self.epochs, idx+1, n_train_batch, loss.item()), flush=True)
        
        train_loss /= n_train_batch
        
        return train_loss
    
    def run_one_epoch_eval(self, epoch):
        # Override
        """
        Validation
        """
        F_bin = self.F_bin
        
        self.model.eval()
        
        valid_loss = 0
        n_valid = len(self.valid_loader.dataset)
        
        with torch.no_grad():
            for idx, (mixture, sources, assignment, threshold_weight) in enumerate(self.valid_loader):
                """
                mixture (batch_size, 1, 2*F_bin, T_bin)
                sources (batch_size, n_sources, F_bin, T_bin)
                assignment (batch_size, n_sources, F_bin, T_bin)
                """
                if self.use_cuda:
                    mixture = mixture.cuda()
                    sources = sources.cuda()
                    assignment = assignment.cuda()
                    threshold_weight = threshold_weight.cuda()
                
                real, imag = mixture[:,:,:F_bin], mixture[:,:,F_bin:]
                mixture_amplitude = torch.sqrt(real**2+imag**2)
                real, imag = sources[:,:,:F_bin], sources[:,:,F_bin:]
                sources_amplitude = torch.sqrt(real**2+imag**2)
                
                output = self.model(mixture_amplitude, assignment=assignment, threshold_weight=threshold_weight)
                loss, _ = self.pit_criterion(output, sources_amplitude, batch_mean=False)
                loss = loss.sum(dim=0)
                valid_loss += loss.item()
                
                if idx < 5:
                    mixture = mixture[0].cpu() # -> (1, 2*F_bin, T_bin)
                    mixture_amplitude = mixture_amplitude[0].cpu() # -> (n_sources, F_bin, T_bin)
                    estimated_sources_amplitude = output[0].cpu() # -> (n_sources, F_bin, T_bin)
                    ratio = estimated_sources_amplitude / mixture_amplitude
                    real, imag = mixture[:,:F_bin], mixture[:,F_bin:]
                    real, imag = ratio * real, ratio * imag
                    estimated_sources = torch.cat([real, imag], dim=1) # -> (n_sources, 2*F_bin, T_bin)
                    estimated_sources = self.istft(estimated_sources) # -> (n_sources, T)
                    estimated_sources = estimated_sources.detach().cpu().numpy()
                    
                    for source_idx, estimated_source in enumerate(estimated_sources):
                        save_dir = os.path.join(self.sample_dir, "{}".format(idx+1))
                        os.makedirs(save_dir, exist_ok=True)
                        save_path = os.path.join(save_dir, "epoch{}-{}.wav".format(epoch+1,source_idx))
                        norm = np.abs(estimated_source).max()
                        estimated_source = estimated_source / norm
                        write_wav(save_path, signal=estimated_source, sr=self.sr)
        
        valid_loss /= n_valid
        
        return valid_loss
