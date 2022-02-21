import os
import time

import musdb
import museval
import torch
import torchaudio
import torch.nn as nn

from utils.utils import draw_loss_curve
from algorithm.frequency_mask import multichannel_wiener_filter

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
                self.model.module.load_state_dict(config['state_dict'])
            else:
                self.model.load_state_dict(config['state_dict'])

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

        # For save_model
        if hasattr(args, 'save_normalized'):
            self.save_normalized = args.save_normalized
        else:
            self.save_normalized = False

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
                estimated_sources = self.model(mixture)
                loss = self.criterion(estimated_sources, sources, batch_mean=False)
                loss = loss.sum(dim=0)
                valid_loss += loss.item()

                if idx < 5:
                    mixture = mixture[0].squeeze(dim=0).detach().cpu()
                    estimated_sources = estimated_sources[0].detach().cpu()

                    save_dir = os.path.join(self.sample_dir, titles[0])
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, "mixture.wav")

                    if self.save_normalized:
                        norm = torch.abs(mixture).max()
                        mixture = mixture / norm

                    torchaudio.save(save_path, mixture, sample_rate=self.sample_rate, bits_per_sample=BITS_PER_SAMPLE_MUSDB18)

                    save_dir = os.path.join(self.sample_dir, titles[0], "epoch{}".format(epoch + 1))
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

        torch.save(config, model_path)

class TesterBase:
    def __init__(self, model, loader, criterion, args):
        self.loader = loader

        self.model = model

        self.criterion = criterion

        self._reset(args)

    def _reset(self, args):
        self.sample_rate = args.sample_rate
        self.n_sources = args.n_sources

        self.out_dir = args.out_dir

        if self.out_dir is not None:
            self.out_dir = os.path.abspath(args.out_dir)
            os.makedirs(self.out_dir, exist_ok=True)

        self.use_cuda = args.use_cuda

        config = torch.load(args.model_path, map_location=lambda storage, loc: storage)

        if isinstance(self.model, nn.DataParallel):
            self.model.module.load_state_dict(config['state_dict'])
        else:
            self.model.load_state_dict(config['state_dict'])

        # For save_model
        if hasattr(args, 'save_normalized'):
            self.save_normalized = args.save_normalized
        else:
            self.save_normalized = False

    def run(self):
        raise NotImplementedError("Implement `run` in sub-class.")

class EvaluaterBase:
    def __init__(self, args):
        self._reset(args)

    def _reset(self, args):
        self.sample_rate = args.sample_rate
        self.sources = args.sources

        self.musdb18_root = args.musdb18_root

        self.estimates_dir = args.estimates_dir
        self.json_dir = args.json_dir

        if self.json_dir is not None:
            self.json_dir = os.path.abspath(args.json_dir)
            os.makedirs(self.json_dir, exist_ok=True)

        self.use_norbert = args.use_norbert

        if self.use_norbert:
            try:
                import norbert
            except:
                raise ImportError("Cannot import norbert.")

    def run(self):
        mus = musdb.DB(root=self.musdb18_root, subsets='test', is_wav=True)

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

    def apply_multichannel_wiener_filter(self, mixture, estimated_sources_amplitude, channels_first=True, eps=EPS):
        if self.use_norbert:
            estimated_sources = apply_multichannel_wiener_filter_norbert(mixture, estimated_sources_amplitude, channels_first=channels_first, eps=eps)
        else:
            estimated_sources = apply_multichannel_wiener_filter_torch(mixture, estimated_sources_amplitude, channels_first=channels_first, eps=eps)

        return estimated_sources

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
    return multichannel_wiener_filter(mixture, estimated_sources_amplitude, iteration=iteration, channels_first=channels_first, eps=eps)
