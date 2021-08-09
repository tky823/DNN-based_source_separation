import os
import shutil
import subprocess
import time
import uuid

import numpy as np
from mir_eval.separation import bss_eval_sources
import torch
import torchaudio
import torch.nn as nn

from utils.utils import draw_loss_curve
from criterion.pit import pit

MIN_PESQ = -0.5

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
                    for param_group in self.optimizer.param_groups:
                        prev_lr = param_group['lr']
                        lr = 0.5 * prev_lr
                        print("Learning rate: {} -> {}".format(prev_lr, lr))
                        param_group['lr'] = lr
            
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
                    mixture = mixture[0].squeeze(dim=0).detach().cpu()
                    estimated_sources = output[0].detach().cpu()
                    
                    save_dir = os.path.join(self.sample_dir, "{}".format(idx+1))
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, "mixture.wav")
                    norm = torch.abs(mixture).max()
                    mixture = mixture / norm
                    signal = mixture.unsqueeze(dim=0) if mixture.dim() == 1 else mixture
                    torchaudio.save(save_path, signal, sample_rate=self.sr)
                    
                    for source_idx, estimated_source in enumerate(estimated_sources):
                        save_path = os.path.join(save_dir, "epoch{}-{}.wav".format(epoch+1,source_idx+1))
                        norm = torch.abs(estimated_source).max()
                        estimated_source = estimated_source / norm
                        signal = estimated_source.unsqueeze(dim=0) if estimated_source.dim() == 1 else estimated_source
                        torchaudio.save(save_path, signal, sample_rate=self.sr)
        
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
            self.out_dir = os.path.abspath(self.out_dir)
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
        test_sdr_improvement = 0
        test_sir_improvement = 0
        test_sar = 0
        test_pesq = 0
        n_pesq_error = 0
        n_test = len(self.loader.dataset)

        print("ID, Loss, Loss improvement, SDR improvement, SIR improvement, SAR, PESQ", flush=True)

        tmp_dir = os.path.join(os.getcwd(), 'tmp')
        os.makedirs(tmp_dir, exist_ok=True)
        shutil.copy('./PESQ', os.path.join(tmp_dir, 'PESQ'))
        os.chdir(tmp_dir)
        
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
                loss_improvement = loss_mixture.item() - loss.item()
                
                mixture = mixture[0].squeeze(dim=0).cpu() # -> (T,)
                sources = sources[0].cpu() # -> (n_sources, T)
                estimated_sources = output[0].cpu() # -> (n_sources, T)
                perm_idx = perm_idx[0] # -> (n_sources,)
                segment_IDs = segment_IDs[0] # -> (n_sources,)

                repeated_mixture = torch.tile(mixture, (self.n_sources, 1))
                result_estimated = bss_eval_sources(
                    reference_sources=sources.numpy(),
                    estimated_sources=estimated_sources.numpy()
                )
                result_mixed = bss_eval_sources(
                    reference_sources=sources.numpy(),
                    estimated_sources=repeated_mixture.numpy()
                )
        
                sdr_improvement = np.mean(result_estimated[0] - result_mixed[0])
                sir_improvement = np.mean(result_estimated[1] - result_mixed[1])
                sar = np.mean(result_estimated[2])
                
                norm = torch.abs(mixture).max()
                mixture /= norm
                mixture_ID = "+".join(segment_IDs)

                # Generate random number temporary wav file.
                random_ID = str(uuid.uuid4())
                
                if idx < 10 and self.out_dir is not None:
                    mixture_path = os.path.join(self.out_dir, "{}.wav".format(mixture_ID))
                    signal = mixture.unsqueeze(dim=0) if mixture.dim() == 1 else mixture
                    torchaudio.save(mixture_path, signal, sample_rate=self.sr)
                
                for order_idx in range(self.n_sources):
                    source, estimated_source = sources[order_idx], estimated_sources[perm_idx[order_idx]]
                    segment_ID = segment_IDs[order_idx]
                    
                    # Target
                    norm = torch.abs(source).max()
                    source /= norm
                    if idx < 10 and  self.out_dir is not None:
                        source_path = os.path.join(self.out_dir, "{}_{}-target.wav".format(mixture_ID, order_idx + 1))
                        signal = source.unsqueeze(dim=0) if source.dim() == 1 else source
                        torchaudio.save(source_path, signal, sample_rate=self.sr)
                    source_path = "tmp-{}-target_{}.wav".format(order_idx + 1, random_ID)
                    signal = source.unsqueeze(dim=0) if source.dim() == 1 else source
                    torchaudio.save(source_path, signal, sample_rate=self.sr)
                    
                    # Estimated source
                    norm = torch.abs(estimated_source).max()
                    estimated_source /= norm
                    if idx < 10 and  self.out_dir is not None:
                        estimated_path = os.path.join(self.out_dir, "{}_{}-estimated.wav".format(mixture_ID, order_idx + 1))
                        signal = estimated_source.unsqueeze(dim=0) if estimated_source.dim() == 1 else estimated_source
                        torchaudio.save(estimated_path, signal, sample_rate=self.sr)
                    estimated_path = "tmp-{}-estimated_{}.wav".format(order_idx + 1, random_ID)
                    signal = estimated_source.unsqueeze(dim=0) if estimated_source.dim() == 1 else estimated_source
                    torchaudio.save(estimated_path, signal, sample_rate=self.sr)
                
                pesq = 0
                
                for source_idx in range(self.n_sources):
                    source_path = "tmp-{}-target_{}.wav".format(source_idx + 1, random_ID)
                    estimated_path = "tmp-{}-estimated_{}.wav".format(source_idx + 1, random_ID)
                    
                    command = "./PESQ +{} {} {}".format(self.sr, source_path, estimated_path)
                    command += " | grep Prediction | awk '{print $5}'"
                    pesq_output = subprocess.check_output(command, shell=True)
                    pesq_output = pesq_output.decode().strip()
                    
                    if pesq_output == '':
                        # If processing error occurs in PESQ software, it is regarded as PESQ score is -0.5. (minimum of PESQ)
                        n_pesq_error += 1
                        pesq += MIN_PESQ
                    else:
                        pesq += float(pesq_output)
                    
                    subprocess.call("rm {}".format(source_path), shell=True)
                    subprocess.call("rm {}".format(estimated_path), shell=True)
                
                pesq /= self.n_sources
                print("{}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}".format(mixture_ID, loss.item(), loss_improvement, sdr_improvement, sir_improvement, sar, pesq), flush=True)
                
                test_loss += loss.item()
                test_loss_improvement += loss_improvement
                test_sdr_improvement += sdr_improvement
                test_sir_improvement += sir_improvement
                test_sar += sar
                test_pesq += pesq
        
        test_loss /= n_test
        test_loss_improvement /= n_test
        test_sdr_improvement /= n_test
        test_sir_improvement /= n_test
        test_sar /= n_test
        test_pesq /= n_test

        os.chdir("../") # back to the original directory
            
        print("Loss: {:.3f}, loss improvement: {:3f}, SDR improvement: {:3f}, SIR improvement: {:3f}, SAR: {:3f}, PESQ: {:.3f}".format(test_loss, test_loss_improvement, test_sdr_improvement, test_sir_improvement, test_sar, test_pesq))
        print("Evaluation of PESQ returns error {} times".format(n_pesq_error))

class AttractorTrainer(Trainer):
    def __init__(self, model, loader, criterion, optimizer, args):
        self.train_loader, self.valid_loader = loader['train'], loader['valid']
        
        self.model = model
        
        self.criterion = criterion
        self.optimizer = optimizer
        
        self._reset(args)
    
    def _reset(self, args):
        # Override
        super()._reset(args)
        
        self.n_bins = args.n_bins
        self.fft_size, self.hop_size = args.fft_size, args.hop_size

        if args.window_fn:
            if args.window_fn == 'hann':
                self.window = torch.hann_window(self.fft_size)
            else:
                raise ValueError("Invalid argument.")
        else:
            self.window = None
        
        self.normalize = args.normalize # TODO: check
    
    def run_one_epoch_train(self, epoch):
        # Override
        """
        Training
        """
        self.model.train()
        
        train_loss = 0
        n_train_batch = len(self.train_loader)
        
        for idx, (mixture, sources, assignment, threshold_weight) in enumerate(self.train_loader):
            if self.use_cuda:
                mixture = mixture.cuda()
                sources = sources.cuda()
                assignment = assignment.cuda()
                threshold_weight = threshold_weight.cuda()
            
            mixture_amplitude = torch.abs(mixture)
            sources_amplitude = torch.abs(sources)
            
            estimated_sources_amplitude = self.model(mixture_amplitude, assignment=assignment, threshold_weight=threshold_weight, n_sources=sources.size(1))
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
        n_sources = self.n_sources
        
        self.model.eval()
        
        valid_loss = 0
        n_valid = len(self.valid_loader.dataset)
        
        with torch.no_grad():
            for idx, (mixture, sources, assignment, threshold_weight) in enumerate(self.valid_loader):
                """
                mixture (batch_size, 1, 2*n_bins, n_frames)
                sources (batch_size, n_sources, 2*n_bins, n_frames)
                assignment (batch_size, n_sources, n_bins, n_frames)
                threshold_weight (batch_size, n_bins, n_bins)
                """
                if self.use_cuda:
                    mixture = mixture.cuda()
                    sources = sources.cuda()
                    threshold_weight = threshold_weight.cuda()
                    assignment = assignment.cuda()
                
                mixture_amplitude = torch.abs(mixture)
                sources_amplitude = torch.abs(sources)
                
                output = self.model(mixture_amplitude, assignment=None, threshold_weight=threshold_weight, n_sources=n_sources)
                # At the test phase, assignment may be unknown.
                loss, _ = pit(self.criterion, output, sources_amplitude, batch_mean=False)
                loss = loss.sum(dim=0)
                valid_loss += loss.item()
                
                if idx < 5:
                    mixture = mixture[0].cpu() # -> (1, n_bins, n_frames)
                    mixture_amplitude = mixture_amplitude[0].cpu() # -> (1, n_bins, n_frames)
                    estimated_sources_amplitude = output[0].cpu() # -> (n_sources, n_bins, n_frames)
                    ratio = estimated_sources_amplitude / mixture_amplitude
                    estimated_sources = ratio * mixture
                    estimated_sources = torch.istft(estimated_sources, n_fft=self.fft_size, hop_length=self.hop_size, normalized=self.normalize, window=self.window) # -> (n_sources, T)
                    
                    mixture = torch.istft(mixture, n_fft=self.fft_size, hop_length=self.hop_size, normalized=self.normalize, window=self.window).squeeze(dim=0) # -> (T,)
                    
                    save_dir = os.path.join(self.sample_dir, "{}".format(idx+1))
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, "mixture.wav")
                    norm = torch.abs(mixture).max()
                    mixture = mixture / norm
                    signal = mixture.unsqueeze(dim=0) if mixture.dim() == 1 else mixture
                    torchaudio.save(save_path, signal, sample_rate=self.sr)
                    
                    for source_idx, estimated_source in enumerate(estimated_sources):
                        save_path = os.path.join(save_dir, "epoch{}-{}.wav".format(epoch+1,source_idx+1))
                        norm = torch.abs(estimated_source).max()
                        estimated_source = estimated_source / norm
                        signal = estimated_source.unsqueeze(dim=0) if estimated_source.dim() == 1 else estimated_source
                        torchaudio.save(save_path, signal, sample_rate=self.sr)
        
        valid_loss /= n_valid
        
        return valid_loss

class AttractorTester(Tester):
    def __init__(self, model, loader, pit_criterion, args):
        self.loader = loader
        
        self.model = model
        
        self.pit_criterion = pit_criterion
        
        self._reset(args)
    
    def _reset(self, args):
        # Override
        super()._reset(args)

        self.n_bins = args.n_bins
        self.fft_size, self.hop_size = args.fft_size, args.hop_size

        if args.window_fn:
            if args.window_fn == 'hann':
                self.window = torch.hann_window(self.fft_size)
            else:
                raise ValueError("Invalid argument.")
        else:
            self.window = None
        
        self.normalize = args.normalize # TODO: check
    
    def run(self):
        n_sources = self.n_sources
        
        self.model.eval()
        
        test_loss = 0
        test_pesq = 0
        n_pesq_error = 0
        n_test = len(self.loader.dataset)
        
        with torch.no_grad():
            for idx, (mixture, sources, ideal_mask, threshold_weight, T, segment_IDs) in enumerate(self.loader):
                """
                mixture (1, 1, n_bins, n_frames, 2)
                sources (1, n_sources, n_bins, n_frames, 2)
                assignment (1, n_sources, n_bins, n_frames)
                threshold_weight (1, n_bins, n_frames)
                T (1,)
                """
                if self.use_cuda:
                    mixture = mixture.cuda()
                    sources = sources.cuda()
                    ideal_mask = ideal_mask.cuda()
                    threshold_weight = threshold_weight.cuda()
                
                mixture_amplitude = torch.abs(mixture) # -> (1, 1, n_bins, n_frames)
                sources_amplitude = torch.abs(sources)
                
                output = self.model(mixture_amplitude, assignment=None, threshold_weight=threshold_weight, n_sources=n_sources)
                loss, perm_idx = self.pit_criterion(output, sources_amplitude, batch_mean=False)
                loss = loss.sum(dim=0)
                
                mixture = mixture[0].cpu()
                sources = sources[0].cpu()
    
                mixture_amplitude = mixture_amplitude[0].cpu() # -> (1, n_bins, n_frames)
                estimated_sources_amplitude = output[0].cpu() # -> (n_sources, n_bins, n_frames)
                ratio = estimated_sources_amplitude / mixture_amplitude
                estimated_sources = ratio * mixture # -> (n_sources, n_bins, n_frames)

                perm_idx = perm_idx[0] # -> (n_sources,)
                T = T[0]  # -> ()
                segment_IDs = segment_IDs[0] # -> (n_sources,)
                mixture = torch.istft(mixture, n_fft=self.fft_size, hop_length=self.hop_size, normalized=self.normalize, window=self.window, length=T).squeeze(dim=0) # -> (T,)
                sources = torch.istft(sources, n_fft=self.fft_size, hop_length=self.hop_size, normalized=self.normalize, window=self.window, length=T) # -> (n_sources, T)
                estimated_sources = torch.istft(estimated_sources, n_fft=self.fft_size, hop_length=self.hop_size, normalized=self.normalize, window=self.window, length=T) # -> (n_sources, T)
                
                norm = torch.abs(mixture).max()
                mixture /= norm
                mixture_ID = "+".join(segment_IDs)

                # Generate random number temporary wav file.
                random_ID = str(uuid.uuid4())
                    
                if idx < 10 and self.out_dir is not None:
                    mixture_path = os.path.join(self.out_dir, "{}.wav".format(mixture_ID))
                    signal = mixture.unsqueeze(dim=0) if mixture.dim() == 1 else mixture
                    torchaudio.save(mixture_path, signal, sample_rate=self.sr)
                    
                for order_idx in range(self.n_sources):
                    source, estimated_source = sources[order_idx], estimated_sources[perm_idx[order_idx]]
                    segment_ID = segment_IDs[order_idx]
                    
                    # Target
                    norm = torch.abs(source).max()
                    source /= norm
                    if idx < 10 and  self.out_dir is not None:
                        source_path = os.path.join(self.out_dir, "{}_{}-target.wav".format(mixture_ID, order_idx))
                        signal = source.unsqueeze(dim=0) if source.dim() == 1 else source
                        torchaudio.save(source_path, signal, sample_rate=self.sr)
                    source_path = "tmp-{}-target_{}.wav".format(order_idx, random_ID)
                    signal = source.unsqueeze(dim=0) if source.dim() == 1 else source
                    torchaudio.save(source_path, signal, sample_rate=self.sr)
                    
                    # Estimated source
                    norm = torch.abs(estimated_source).max()
                    estimated_source /= norm
                    if idx < 10 and  self.out_dir is not None:
                        estimated_path = os.path.join(self.out_dir, "{}_{}-estimated.wav".format(mixture_ID, order_idx))
                        signal = estimated_source.unsqueeze(dim=0) if estimated_source.dim() == 1 else estimated_source
                        torchaudio.save(estimated_path, signal, sample_rate=self.sr)
                    estimated_path = "tmp-{}-estimated_{}.wav".format(order_idx, random_ID)
                    signal = estimated_source.unsqueeze(dim=0) if estimated_source.dim() == 1 else estimated_source
                    torchaudio.save(estimated_path, signal, sample_rate=self.sr)
                
                pesq = 0
                    
                for source_idx in range(self.n_sources):
                    source_path = "tmp-{}-target_{}.wav".format(source_idx, random_ID)
                    estimated_path = "tmp-{}-estimated_{}.wav".format(source_idx, random_ID)
                    
                    command = "./PESQ +{} {} {}".format(self.sr, source_path, estimated_path)
                    command += " | grep Prediction | awk '{print $5}'"
                    pesq_output = subprocess.check_output(command, shell=True)
                    pesq_output = pesq_output.decode().strip()
                    
                    if pesq_output == '':
                        # If processing error occurs in PESQ software, it is regarded as PESQ score is -0.5. (minimum of PESQ)
                        n_pesq_error += 1
                        pesq += -0.5
                    else:
                        pesq += float(pesq_output)
                    
                    subprocess.call("rm {}".format(source_path), shell=True)
                    subprocess.call("rm {}".format(estimated_path), shell=True)
                
                pesq /= self.n_sources
                print("{}, {:.3f}, {:.3f}".format(mixture_ID, loss.item(), pesq), flush=True)
                
                test_loss += loss.item()
                test_pesq += pesq
        
        test_loss /= n_test
        test_pesq /= n_test
                
        print("Loss: {:.3f}, PESQ: {:.3f}".format(test_loss, test_pesq))
        print("Evaluation of PESQ returns error {} times".format(n_pesq_error))

class AnchoredAttractorTrainer(AttractorTrainer):
    def __init__(self, model, loader, criterion, optimizer, args):
        super().__init__(model, loader, criterion, optimizer, args)
    
    def run_one_epoch_train(self, epoch):
        # Override
        """
        Training
        """
        self.model.train()
        
        train_loss = 0
        n_train_batch = len(self.train_loader)
        
        for idx, (mixture, sources, threshold_weight) in enumerate(self.train_loader):
            if self.use_cuda:
                mixture = mixture.cuda()
                sources = sources.cuda()
                threshold_weight = threshold_weight.cuda()
            
            mixture_amplitude = torch.abs(mixture)
            sources_amplitude = torch.abs(sources)
            
            estimated_sources_amplitude = self.model(mixture_amplitude, threshold_weight=threshold_weight, n_sources=sources.size(1))
            loss = self.criterion(estimated_sources_amplitude, sources_amplitude)
            
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
        n_sources = self.n_sources
        
        self.model.eval()
        
        valid_loss = 0
        n_valid = len(self.valid_loader.dataset)
        
        with torch.no_grad():
            for idx, (mixture, sources, threshold_weight) in enumerate(self.valid_loader):
                """
                mixture (batch_size, 1, n_bins, n_frames)
                sources (batch_size, n_sources, n_bins, n_frames)
                threshold_weight (batch_size, n_bins, n_frames)
                """
                if self.use_cuda:
                    mixture = mixture.cuda()
                    sources = sources.cuda()
                    threshold_weight = threshold_weight.cuda()
                
                mixture_amplitude = torch.abs(mixture)
                sources_amplitude = torch.abs(sources)
                
                output = self.model(mixture_amplitude, threshold_weight=threshold_weight, n_sources=n_sources)
                # At the test phase, assignment may be unknown.
                loss, _ = pit(self.criterion, output, sources_amplitude, batch_mean=False)
                loss = loss.sum(dim=0)
                valid_loss += loss.item()
                
                if idx < 5:
                    mixture = mixture[0].cpu() # -> (1, n_bins, n_frames)
                    mixture_amplitude = mixture_amplitude[0].cpu() # -> (1, n_bins, n_frames)
                    estimated_sources_amplitude = output[0].cpu() # -> (n_sources, n_bins, n_frames)
                    ratio = estimated_sources_amplitude / mixture_amplitude
                    estimated_sources = ratio * mixture
                    estimated_sources = torch.istft(estimated_sources, n_fft=self.fft_size, hop_length=self.hop_size, normalized=self.normalize, window=self.window) # -> (n_sources, T)
                    estimated_sources = estimated_sources.cpu()
                    
                    mixture = torch.istft(mixture, n_fft=self.fft_size, hop_length=self.hop_size, normalized=self.normalize, window=self.window).squeeze(dim=0) # -> (T,)
                    
                    save_dir = os.path.join(self.sample_dir, "{}".format(idx+1))
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, "mixture.wav")
                    norm = torch.abs(mixture).max()
                    mixture = mixture / norm
                    signal = mixture.unsqueeze(dim=0) if mixture.dim() == 1 else mixture
                    torchaudio.save(save_path, signal, sample_rate=self.sr)
                    
                    for source_idx, estimated_source in enumerate(estimated_sources):
                        save_path = os.path.join(save_dir, "epoch{}-{}.wav".format(epoch+1, source_idx+1))
                        norm = torch.abs(estimated_source).max()
                        estimated_source = estimated_source / norm
                        signal = estimated_source.unsqueeze(dim=0) if estimated_source.dim() == 1 else estimated_source
                        torchaudio.save(save_path, signal, sample_rate=self.sr)
        
        valid_loss /= n_valid
        
        return valid_loss

class ORPITTrainer(Trainer):
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
        
        self.use_cuda = args.use_cuda
        
        if args.continue_from:
            package = torch.load(args.continue_from, map_location=lambda storage, loc: storage)

            self.start_epoch = package['epoch']
            self.train_loss[:self.start_epoch] = package['train_loss'][:self.start_epoch]
            
            if isinstance(self.model, nn.DataParallel):
                self.model.module.load_state_dict(package['state_dict'])
            else:
                self.model.load_state_dict(package['state_dict'])
            
            self.optimizer.load_state_dict(package['optim_dict'])
        else:
            # TODO: redundant? last.pth never exists
            model_path = os.path.join(self.model_dir, "last.pth")
            
            if os.path.exists(model_path):
                if args.overwrite:
                    print("Overwrite models.")
                else:
                    raise ValueError("{} already exists. If you continue to run, set --overwrite to be True.".format(model_path))
            
            self.start_epoch = 0
    
    def run(self):
        for epoch in range(self.start_epoch, self.epochs):
            start = time.time()
            train_loss = self.run_one_epoch(epoch)
            end = time.time()
            
            print("[Epoch {}/{}] loss (train): {:.5f}, {:.3f} [sec]".format(epoch+1, self.epochs, train_loss, end - start), flush=True)
            
            self.train_loss[epoch] = train_loss
            
            model_path = os.path.join(self.model_dir, "last.pth")
            self.save_model(epoch, model_path)
            
            save_path = os.path.join(self.loss_dir, "loss.png")
            draw_loss_curve(train_loss=self.train_loss[:epoch+1], save_path=save_path)
    
    def run_one_epoch(self, epoch):
        """
        Training
        """
        train_loss = self.run_one_epoch_train(epoch)
        _ = self.run_one_epoch_eval(epoch)

        return train_loss
    
    def run_one_epoch_eval(self, epoch):
        """
        Validation
        """
        self.model.eval()

        with torch.no_grad():
            for idx, (mixture, sources) in enumerate(self.valid_loader):
                if self.use_cuda:
                    mixture = mixture.cuda()
                    sources = sources.cuda()
                
                output_one_and_rest = self.model(mixture)
                output_one = output_one_and_rest[:,0:1]
                output_rest = output_one_and_rest[:,1:]
                output = output_one

                for source_idx in range(1, self.n_sources-1):
                    output_one_and_rest = self.model(output_rest)
                    output_one = output_one_and_rest[:,0:1]
                    output_rest = output_one_and_rest[:,1:]
                    output = torch.cat([output, output_one], dim=1)
                
                output = torch.cat([output, output_rest], dim=1)
                
                if idx < 5:
                    mixture = mixture[0].squeeze(dim=0).detach().cpu()
                    estimated_sources = output[0].detach().cpu()
                    
                    save_dir = os.path.join(self.sample_dir, "{}".format(idx+1))
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, "mixture.wav")
                    norm = torch.abs(mixture).max()
                    mixture = mixture / norm
                    signal = mixture.unsqueeze(dim=0) if mixture.dim() == 1 else mixture
                    torchaudio.save(save_path, signal, sample_rate=self.sr)
                    
                    for source_idx, estimated_source in enumerate(estimated_sources):
                        save_path = os.path.join(save_dir, "epoch{}-{}.wav".format(epoch+1, source_idx+1))
                        norm = torch.abs(estimated_source).max()
                        estimated_source = estimated_source / norm
                        signal = estimated_source.unsqueeze(dim=0) if estimated_source.dim() == 1 else estimated_source
                        torchaudio.save(save_path, signal, sample_rate=self.sr)
        
        return -1
    
    def save_model(self, epoch, model_path='./tmp.pth'):
        if isinstance(self.model, nn.DataParallel):
            package = self.model.module.get_package()
            package['state_dict'] = self.model.module.state_dict()
        else:
            package = self.model.get_package()
            package['state_dict'] = self.model.state_dict()
            
        package['optim_dict'] = self.optimizer.state_dict()
        
        package['epoch'] = epoch + 1
        package['train_loss'] = self.train_loss
        
        torch.save(package, model_path)