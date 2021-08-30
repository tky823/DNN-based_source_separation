import os
import time
import math

import musdb
import museval
import torch
import torchaudio
import torch.nn as nn

from utils.utils import draw_loss_curve

BITS_PER_SAMPLE_MUSDB18 = 16
EPS = 1e-12

class TrainerBase:
    def __init__(self, model, loader, criterion, optimizer, args):
        self.train_loader, self.valid_loader = loader['train'], loader['valid']
        
        self.model = model
        
        self.criterion = criterion
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
            
            self.prev_loss = valid_loss
            
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

class TesterBase:
    def __init__(self, model, loader, criterion, args):
        self.loader = loader
        
        self.model = model
        
        self.criterion = criterion
        
        self._reset(args)
        
    def _reset(self, args):
        self.sr = args.sr
        self.n_sources = args.n_sources
        
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
        raise NotImplementedError("Implement `run` in sub-class.")

class EvaluaterBase:
    def __init__(self, args):
        self._reset(args)
    
    def _reset(self, args):
        self.target = [
            'drums', 'bass', 'other', 'vocals', 'accompaniment'
        ]
        self.mus = musdb.DB(root=args.musdb18_root, subsets="test", is_wav=args.is_wav)
        self.estimated_mus = musdb.DB(root=args.estimated_musdb18_root, subsets="test", is_wav=True)
        self.json_dir = args.json_dir

        os.makedirs(self.json_dir, exist_ok=True)
    
    def run(self):
        results = museval.EvalStore(frames_agg='median', tracks_agg='median')

        for track, estimated_track in zip(self.mus.tracks, self.estimated_mus.tracks):
            scores = self.run_one_track(track, estimated_track)
            results.add_track(scores)
        
        print(results)
    
    def run_one_track(self, track, estimated_track):
        estimates = {}

        for _target in self.target:
            estimates[_target] = estimated_track.targets[_target].audio
        
        scores = museval.eval_mus_track(
            track, estimates, output_dir=self.json_dir
        )

        return scores

def apply_multichannel_wiener_filter_norbert(mixture, estimated_sources_amplitude, iteration=1, channels_first=True, eps=EPS):
    """
    Args:
        mixture <torch.Tensor>: (1, n_channels, n_bins, n_frames) or (n_channels, n_bins, n_frames), complex tensor
        estimated_sources_amplitude <torch.Tensor>: (n_sources, n_channels, n_bins, n_frames), real (nonnegative) tensor
    Returns:
        estimated_sources <torch.Tensor>: (n_sources, n_channels, n_bins, n_frames), complex tensor
    """
    import norbert

    assert channels_first, "`channels_first` is expected True, but given {}".format(channels_first)

    n_dims = mixture.dim()

    if n_dims == 4:
        mixture = mixture.squeeze(dim=0)
    elif n_dims != 3:
        raise ValueError("mixture.dim() is expected 3 or 4, but given {}.".format(mixture.dim()))

    assert estimated_sources_amplitude.dim() == 4, "estimated_sources_amplitude.dim() is expected 4, but given {}.".format(estimated_sources_amplitude.dim())

    device = mixture.device
    dtype = mixture.dtype

    mixture = mixture.detach().cpu().numpy()
    estimated_sources_amplitude = estimated_sources_amplitude.detach().cpu().numpy()

    mixture = mixture.transpose(2, 1, 0)
    estimated_sources_amplitude = estimated_sources_amplitude.transpose(3, 2, 1, 0)
    estimated_sources = norbert.wiener(estimated_sources_amplitude, mixture, iterations=iteration, eps=eps)
    estimated_sources = estimated_sources.transpose(3, 2, 1, 0)
    estimated_sources = torch.from_numpy(estimated_sources).to(device, dtype)

    return estimated_sources

def apply_multichannel_wiener_filter_torch(mixture, estimated_sources_amplitude, iteration=1, channels_first=True, eps=EPS):
    """
    Multichannel Wiener filter.
    Implementation is based on norbert package.
    Args:
        mixture <torch.Tensor>: Complex tensor with shape of (1, n_channels, n_bins, n_frames) or (n_channels, n_bins, n_frames) or (batch_size, 1, n_channels, n_bins, n_frames) or (batch_size, n_channels, n_bins, n_frames)
        estimated_sources_amplitude <torch.Tensor>: (n_sources, n_channels, n_bins, n_frames) or (batch_size, n_sources, n_channels, n_bins, n_frames)
        iteration <int>: Iteration of EM algorithm updates
        channels_first <bool>: Only supports True
        eps <float>: small value for numerical stability
    """
    assert channels_first, "`channels_first` is expected True, but given {}".format(channels_first)

    n_dims = estimated_sources_amplitude.dim()
    n_dims_mixture = mixture.dim()

    if n_dims == 4:
        """
        Shape of mixture is (1, n_channels, n_bins, n_frames) or (n_channels, n_bins, n_frames)
        """
        if n_dims_mixture == 4:
            mixture = mixture.squeeze(dim=1) # (n_channels, n_bins, n_frames)
        elif n_dims_mixture != 3:
            raise ValueError("mixture.dim() is expected 3 or 4, but given {}.".format(mixture.dim()))
        
        # Use soft mask
        ratio = estimated_sources_amplitude / (estimated_sources_amplitude.sum(dim=0) + eps)
        estimated_sources = ratio * mixture

        norm = max(1, torch.abs(mixture).max() / 10)
        mixture, estimated_sources = mixture / norm, estimated_sources / norm

        estimated_sources = update_em(mixture, estimated_sources, iteration, eps=eps)
        estimated_sources = norm * estimated_sources
    elif n_dims == 5:
        """
        Shape of mixture is (batch_size, 1, n_channels, n_bins, n_frames) or (batch_size, n_channels, n_bins, n_frames)
        """
        if n_dims_mixture == 5:
            mixture = mixture.squeeze(dim=1) # (batch_size, n_channels, n_bins, n_frames)
        elif n_dims_mixture != 4:
            raise ValueError("mixture.dim() is expected 4 or 5, but given {}.".format(mixture.dim()))
        
        estimated_sources = []

        for _mixture, _estimated_sources_amplitude in zip(mixture, estimated_sources_amplitude):
            # Use soft mask
            ratio = _estimated_sources_amplitude / (_estimated_sources_amplitude.sum(dim=0) + eps)
            _estimated_sources = ratio * _mixture

            norm = max(1, torch.abs(_mixture).max() / 10)
            _mixture, _estimated_sources = _mixture / norm, _estimated_sources / norm

            _estimated_sources = update_em(_mixture, _estimated_sources, iteration, eps=eps)
            _estimated_sources = norm * _estimated_sources

            estimated_sources.append(_estimated_sources.unsqueeze(dim=0))
        
        estimated_sources = torch.cat(estimated_sources, dim=0)
    else:
        raise ValueError("estimated_sources_amplitude.dim() is expected 4 or 5, but given {}.".format(estimated_sources_amplitude.dim()))

    return estimated_sources

def update_em(mixture, estimated_sources, iterations=1, source_parallel=False, eps=EPS):
    """
    Args:
        mixture: (n_channels, n_bins, n_frames)
        estimated_sources: (n_sources, n_channels, n_bins, n_frames)
    Returns
        estiamted_sources: (n_sources, n_channels, n_bins, n_frames)
    """
    n_sources, n_channels, _, _ = estimated_sources.size()

    for iteration_idx in range(iterations):
        v, R = [], []
        Cxx = 0

        if source_parallel:
            v, R = get_stats(estimated_sources, eps=eps) # (n_sources, n_bins, n_frames), (n_sources, n_bins, n_channels, n_channels)
            Cxx = torch.sum(v.unsqueeze(dim=4) * R, dim=0) # (n_bins, n_frames, n_channels, n_channels)
        else:
            for source_idx in range(n_sources):
                y_n = estimated_sources[source_idx] # (n_channels, n_bins, n_frames)
                v_n, R_n = get_stats(y_n, eps=eps) # (n_bins, n_frames), (n_bins, n_channels, n_channels)
                Cxx = Cxx + v_n.unsqueeze(dim=2).unsqueeze(dim=3) * R_n.unsqueeze(dim=1) # (n_bins, n_frames, n_channels, n_channels)
                v.append(v_n.unsqueeze(dim=0))
                R.append(R_n.unsqueeze(dim=0))
        
            v, R = torch.cat(v, dim=0), torch.cat(R, dim=0) # (n_sources, n_bins, n_frames), (n_sources, n_bins, n_channels, n_channels)
       
        v, R = v.unsqueeze(dim=3), R.unsqueeze(dim=2) # (n_sources, n_bins, n_frames, 1), (n_sources, n_bins, 1, n_channels, n_channels)

        inv_Cxx = torch.linalg.inv(Cxx + math.sqrt(eps) * torch.eye(n_channels)) # (n_bins, n_frames, n_channels, n_channels)

        if source_parallel:
            gain = v.unsqueeze(dim=4) * torch.sum(R.unsqueeze(dim=5) * inv_Cxx.unsqueeze(dim=2), dim=4) # (n_sources, n_bins, n_frames, n_channels, n_channels)
            gain = gain.permute(0, 3, 4, 1, 2) # (n_sources, n_channels, n_channels, n_bins, n_frames)
            estimated_sources = torch.sum(gain * mixture, dim=2) # (n_sources, n_channels, n_bins, n_frames)
        else:
            estimated_sources = []

            for source_idx in range(n_sources):
                v_n, R_n = v[source_idx], R[source_idx] # (n_bins, n_frames, 1), (n_bins, 1, n_channels, n_channels)

                gain_n = v_n.unsqueeze(dim=3) * torch.sum(R_n.unsqueeze(dim=4) * inv_Cxx.unsqueeze(dim=2), dim=3) # (n_bins, n_frames, n_channels, n_channels)
                gain_n = gain_n.permute(2, 3, 0, 1) # (n_channels, n_channels, n_bins, n_frames)
                estimated_source = torch.sum(gain_n * mixture, dim=1) # (n_channels, n_bins, n_frames)
                estimated_sources.append(estimated_source.unsqueeze(dim=0))
            
            estimated_sources = torch.cat(estimated_sources, dim=0) # (n_sources, n_channels, n_bins, n_frames)

    return estimated_sources

def get_stats(spectrogram, eps=EPS):
    """
    Compute empirical parameters of local gaussian model.
    Args:
        spectrogram <torch.Tensor>: (n_mics, n_bins, n_frames) or (n_sources, n_mics, n_bins, n_frames)
    Returns:
        psd <torch.Tensor>: (n_bins, n_frames) or (n_sources, n_bins, n_frames)
        covariance <torch.Tensor>: (n_bins, n_frames, n_mics, n_mics) or (n_sources, n_bins, n_frames, n_mics, n_mics)
    """
    n_dims = spectrogram.dim()

    if n_dims == 3:
        psd = torch.mean(torch.abs(spectrogram)**2, dim=0) # (n_bins, n_frames)
        covariance = spectrogram.unsqueeze(dim=1) * spectrogram.unsqueeze(dim=0).conj() # (n_mics, n_mics, n_bins, n_frames)
        covariance = covariance.sum(dim=3) # (n_mics, n_mics, n_bins)
        denominator = psd.sum(dim=1) + eps # (n_bins,)

        covariance = covariance / denominator # (n_mics, n_mics, n_bins, n_frames)
        covariance = covariance.permute(2, 0, 1) # (n_bins, n_mics, n_mics)
    elif n_dims == 4:
        psd = torch.mean(torch.abs(spectrogram)**2, dim=1) # (n_sources, n_bins, n_frames)
        covariance = spectrogram.unsqueeze(dim=2) * spectrogram.unsqueeze(dim=1).conj() # (n_sources, n_mics, n_mics, n_bins, n_frames)
        covariance = covariance.sum(dim=4) # (n_sources, n_mics, n_mics, n_bins)
        denominator = psd.sum(dim=2) + eps # (n_sources, n_bins)
        
        covariance = covariance / denominator.unsqueeze(dim=1).unsqueeze(dim=2) # (n_sources, n_mics, n_mics, n_bins)
        covariance = covariance.permute(0, 3, 1, 2) # (n_sources, n_bins, n_mics, n_mics)
    else:
        raise ValueError("Invalid dimension of tensor is given.")

    return psd, covariance
