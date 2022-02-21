import os
import time

import musdb
import museval
import torch
import torchaudio
import torch.nn as nn

from utils.utils import draw_loss_curve
from transforms.stft import istft
from driver import apply_multichannel_wiener_filter_norbert, apply_multichannel_wiener_filter_torch
from driver import TrainerBase, TesterBase

BITS_PER_SAMPLE_MUSDB18 = 16
EPS = 1e-12

class AdhocTrainer(TrainerBase):
    def __init__(self, model, loader, criterion, optimizer, args):
        self.train_loader, self.valid_loader = loader['train'], loader['valid']

        self.model = model

        self.criterion = criterion
        self.optimizer = optimizer

        self._reset(args)

    def _reset(self, args):
        # Override
        self.sample_rate = args.sample_rate

        self.n_fft, self.hop_length = args.n_fft, args.hop_length
        self.window = self.valid_loader.dataset.window
        self.normalize = self.valid_loader.dataset.normalize

        self.max_norm = args.max_norm

        self.model_dir = args.model_dir
        self.loss_dir = args.loss_dir
        self.sample_dir = args.sample_dir

        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.loss_dir, exist_ok=True)
        os.makedirs(self.sample_dir, exist_ok=True)

        self.epochs = args.epochs
        self.train_loss, self.valid_loss = torch.empty(self.epochs), torch.empty(self.epochs)

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
                self.no_improvement += 1
                if self.no_improvement >= 10:
                    for param_group in self.optimizer.param_groups:
                        prev_lr = param_group['lr']
                        lr = 0.5 * prev_lr
                        print("Learning rate: {} -> {}".format(prev_lr, lr))
                        param_group['lr'] = lr

            self.prev_loss = valid_loss

            model_path = os.path.join(self.model_dir, "last.pth")
            self.save_model(epoch, model_path)

            save_path = os.path.join(self.loss_dir, "loss.png")
            draw_loss_curve(train_loss=self.train_loss[:epoch+1], valid_loss=self.valid_loss[:epoch+1], save_path=save_path)

    def run_one_epoch_train(self, epoch):
        # Override
        """
            Training
        """
        self.model.train()

        train_loss = 0
        n_train_batch = len(self.train_loader)

        for idx, (mixture, source) in enumerate(self.train_loader):
            if self.use_cuda:
                mixture = mixture.cuda()
                source = source.cuda()

            mixture_amplitude = torch.abs(mixture)
            source_amplitude = torch.abs(source)

            estimated_sources_amplitude = self.model(mixture_amplitude)

            loss = self.criterion(estimated_sources_amplitude, source_amplitude)

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
        # Override
        """
            Validation
        """
        self.model.eval()

        valid_loss = 0
        n_valid = len(self.valid_loader.dataset)

        with torch.no_grad():
            for idx, (mixture, source, name) in enumerate(self.valid_loader):
                """
                    mixture: (batch_size, n_mics, n_bins, n_frames)
                    sources: (batch_size, n_mics, n_bins, n_frames)
                    name <list<str>>: Artist and title of song
                """
                if self.use_cuda:
                    mixture = mixture.cuda()
                    source = source.cuda()

                mixture_amplitude = torch.abs(mixture)
                source_amplitude = torch.abs(source)

                estimated_source_amplitude = self.model(mixture_amplitude)
                loss = self.criterion(estimated_source_amplitude, source_amplitude, batch_mean=False)
                loss = loss.mean(dim=0)
                valid_loss += loss.item()

                if idx < 5:
                    ratio = estimated_source_amplitude / torch.clamp(mixture_amplitude, min=EPS)
                    estimated_source = ratio * mixture # -> (batch_size, n_mics, n_bins, n_frames)

                    mixture_channels = mixture.size()[:-2] # -> (batch_size, n_mics)
                    estimated_source_channels = estimated_source.size()[:-2] # -> (batch_size, n_mics)
                    mixture = mixture.view(-1, *mixture.size()[-2:]) # -> (batch_size * n_mics, n_bins, n_frames)
                    estimated_source = estimated_source.view(-1, *estimated_source.size()[-2:]) # -> (batch_size * n_mics, n_bins, n_frames)

                    mixture, estimated_source = mixture.cpu(), estimated_source.cpu()
                    mixture = istft(mixture, self.n_fft, hop_length=self.hop_length, window=self.window, normalized=self.normalize, return_complex=False) # -> (n_mics, T_segment)
                    estimated_source = istft(estimated_source, self.n_fft, hop_length=self.hop_length, window=self.window, normalized=self.normalize, return_complex=False) # -> (n_mics, T_segment)

                    mixture = mixture.view(*mixture_channels, -1) # -> (batch_size, n_mics, T_segment)
                    estimated_source = estimated_source.view(*estimated_source_channels, -1) # -> (batch_size, n_mics, T_segment)

                    batch_size, n_mics, T_segment = mixture.size()

                    mixture = mixture.permute(1, 0, 2) # -> (n_mics, batch_size, T_segment)
                    mixture = mixture.reshape(n_mics, batch_size * T_segment)

                    estimated_source = estimated_source.permute(1, 0, 2) # -> (n_mics, batch_size, T_segment)
                    estimated_source = estimated_source.reshape(n_mics, batch_size * T_segment)

                    save_dir = os.path.join(self.sample_dir, name)

                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, "mixture.wav")
                    signal = mixture.unsqueeze(dim=0) if mixture.dim() == 1 else mixture
                    torchaudio.save(save_path, signal, sample_rate=self.sample_rate, bits_per_sample=BITS_PER_SAMPLE_MUSDB18)

                    save_path = os.path.join(save_dir, "epoch{}.wav".format(epoch + 1))
                    signal = estimated_source.unsqueeze(dim=0) if estimated_source.dim() == 1 else estimated_source
                    torchaudio.save(save_path, signal, sample_rate=self.sample_rate, bits_per_sample=BITS_PER_SAMPLE_MUSDB18)

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

        config['train_loss'], config['valid_loss'] = self.train_loss, self.valid_loss
        config['epoch'] = epoch + 1

        torch.save(config, model_path)

class AdhocTester(TesterBase):
    def __init__(self, model, loader, criterion, args):
        super().__init__(model, loader, criterion, args)

    def _reset(self, args):
        self.sample_rate = args.sample_rate
        self.sources = args.sources

        self.musdb18_root = args.musdb18_root

        self.n_fft, self.hop_length = args.n_fft, args.hop_length    
        self.window = self.loader.dataset.window
        self.normalize = self.loader.dataset.normalize

        self.model_dir = args.model_dir
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
        self.use_norbert = args.use_norbert

        is_data_parallel = isinstance(self.model, nn.DataParallel)

        for target in self.sources:
            model_path = os.path.join(self.model_dir, target, "{}.pth".format(args.model_choice))
            config = torch.load(model_path, map_location=lambda storage, loc: storage)
            if is_data_parallel:
                self.model.module.net[target].load_state_dict(config['state_dict'])
            else:
                self.model.net[target].load_state_dict(config['state_dict'])

        if self.use_norbert:
            try:
                import norbert
            except:
                raise ImportError("Cannot import norbert.")

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

        with torch.no_grad():
            for idx, (mixture, sources, samples, name) in enumerate(self.loader):
                """
                    mixture: (batch_size, 1, n_mics, n_bins, n_frames)
                    sources: (batch_size, n_sources, n_mics, n_bins, n_frames)
                    samples <int>: Length in time domain
                    name <str>: Artist and title of song
                """
                if self.use_cuda:
                    mixture = mixture.cuda()
                    sources = sources.cuda()

                batch_size, n_sources, n_mics, n_bins, n_frames = sources.size()

                mixture_amplitude = torch.abs(mixture)
                sources_amplitude = torch.abs(sources)

                estimated_sources_amplitude = {
                    target: [] for target in self.sources
                }

                # Serial operation
                for _mixture_amplitude in mixture_amplitude:
                    # _mixture_amplitude: (1, n_mics, n_bins, n_frames)
                    for target in self.sources:
                        _estimated_sources_amplitude = self.model(_mixture_amplitude, target=target)
                        estimated_sources_amplitude[target].append(_estimated_sources_amplitude)

                estimated_sources_amplitude = [
                    torch.cat(estimated_sources_amplitude[target], dim=0).unsqueeze(dim=0) for target in self.sources
                ]
                estimated_sources_amplitude = torch.cat(estimated_sources_amplitude, dim=0) # (n_sources, batch_size, n_mics, n_bins, n_frames)
                estimated_sources_amplitude = estimated_sources_amplitude.permute(0, 2, 3, 1, 4)
                estimated_sources_amplitude = estimated_sources_amplitude.reshape(n_sources, n_mics, n_bins, batch_size * n_frames) # (n_sources, n_mics, n_bins, T_pad)

                mixture = mixture.permute(1, 2, 3, 0, 4).reshape(1, n_mics, n_bins, batch_size * n_frames) # (1, n_mics, n_bins, T_pad)
                mixture_amplitude = mixture_amplitude.permute(1, 2, 3, 0, 4).reshape(1, n_mics, n_bins, batch_size * n_frames) # (1, n_mics, n_bins, T_pad)
                sources_amplitude = sources_amplitude.permute(1, 2, 3, 0, 4).reshape(n_sources, n_mics, n_bins, batch_size * n_frames) # (n_sources, n_mics, n_bins, T_pad)

                loss_mixture = self.criterion(mixture_amplitude, sources_amplitude, batch_mean=False) # (n_sources,)
                loss = self.criterion(estimated_sources_amplitude, sources_amplitude, batch_mean=False) # (n_sources,)
                loss_improvement = loss_mixture - loss # (n_sources,)

                mixture = mixture.cpu()
                estimated_sources_amplitude = estimated_sources_amplitude.cpu()

                estimated_sources = self.apply_multichannel_wiener_filter(mixture, estimated_sources_amplitude=estimated_sources_amplitude)
                estimated_sources_channels = estimated_sources.size()[:-2]

                estimated_sources = estimated_sources.view(-1, *estimated_sources.size()[-2:])
                estimated_sources = istft(estimated_sources, self.n_fft, hop_length=self.hop_length, window=self.window, normalized=self.normalize, return_complex=False)
                estimated_sources = estimated_sources.view(*estimated_sources_channels, -1) # -> (n_sources, n_mics, T_pad)

                track_dir = os.path.join(self.estimates_dir, name)
                os.makedirs(track_dir, exist_ok=True)

                for source_idx, target in enumerate(self.sources):
                    estimated_path = os.path.join(track_dir, "{}.wav".format(target))
                    estimated_source = estimated_sources[source_idx, :, :samples] # -> (n_mics, T)
                    signal = estimated_source.unsqueeze(dim=0) if estimated_source.dim() == 1 else estimated_source
                    torchaudio.save(estimated_path, signal, sample_rate=self.sample_rate, bits_per_sample=BITS_PER_SAMPLE_MUSDB18)

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