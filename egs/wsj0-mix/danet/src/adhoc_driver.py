import os
import shutil
import subprocess
import time
import uuid
from collections import OrderedDict

import torch
import torchaudio
import torch.nn as nn

from utils.utils import draw_loss_curve
from utils.bss import bss_eval_sources
from algorithm.clustering import KMeans
from transforms.stft import istft
from driver import TrainerBase, TesterBase
from criterion.pit import pit

BITS_PER_SAMPLE_WSJ0 = 16
MIN_PESQ = -0.5
NO_IMPROVEMENT = 10

class AdhocTrainer(TrainerBase):
    def __init__(self, model, loader, criterion, optimizer, scheduler, args):
        self.train_loader, self.valid_loader = loader['train'], loader['valid']

        self.model = model

        self.criterion = criterion
        self.optimizer, self.scheduler = optimizer, scheduler

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
        self.train_loss, self.valid_loss = torch.empty(self.epochs), torch.empty(self.epochs)

        self.use_cuda = args.use_cuda

        if args.continue_from:
            config = torch.load(args.continue_from, map_location=lambda storage, loc: storage)

            self.start_epoch = config['epoch']

            self.train_loss[:self.start_epoch] = config['train_loss'][:self.start_epoch]
            self.valid_loss[:self.start_epoch] = config['valid_loss'][:self.start_epoch]

            self.best_loss = config['best_loss']
            self.prev_loss = self.valid_loss[self.start_epoch - 1]
            self.no_improvement = config['no_improvement']

            if isinstance(self.model, nn.DataParallel):
                self.model.module.load_state_dict(config['state_dict'])
            else:
                self.model.load_state_dict(config['state_dict'])

            self.optimizer.load_state_dict(config['optim_dict'])
            self.scheduler.load_state_dict(config['scheduler_dict'])
        else:
            model_path = os.path.join(self.model_dir, "best.pth")

            if os.path.exists(model_path):
                if args.overwrite:
                    print("Overwrite models.")
                else:
                    raise ValueError("{} already exists. If you continue to run, set --overwrite to be True.".format(model_path))

            self.start_epoch = 0

            self.best_loss, self.prev_loss = float('infinity'), float('infinity')
            self.no_improvement = 0

        self.n_bins = args.n_bins
        self.n_fft, self.hop_length = args.n_fft, args.hop_length
        self.window = self.train_loader.dataset.window
        self.normalize = self.train_loader.dataset.normalize

        self.target_type = args.target_type
        self.iter_clustering = args.iter_clustering

    def run(self):
        for epoch in range(self.start_epoch, self.epochs):
            start = time.time()
            train_loss, valid_loss = self.run_one_epoch(epoch)
            end = time.time()

            s = "[Epoch {}/{}] loss (train): {:.5f}".format(epoch + 1, self.epochs, train_loss)
            self.train_loss[epoch] = train_loss

            if self.valid_loader is not None:
                s += ", loss (valid): {:.5f}".format(valid_loss)
                self.valid_loss[epoch] = valid_loss

            s += ", {:.3f} [sec]".format(end - start)
            print(s, flush=True)

            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(valid_loss)
            else:
                self.scheduler.step()

            if self.valid_loader is not None:
                if valid_loss < self.best_loss:
                    self.best_loss = valid_loss
                    self.no_improvement = 0
                    model_path = os.path.join(self.model_dir, "best.pth")
                    self.save_model(epoch, model_path)
                else:
                    if valid_loss >= self.prev_loss:
                        self.no_improvement += 1
                        if self.no_improvement >= NO_IMPROVEMENT:
                            print("Stop training")
                            break
                    else:
                        self.no_improvement = 0

                self.prev_loss = valid_loss

            model_path = os.path.join(self.model_dir, "last.pth")
            self.save_model(epoch, model_path)

            save_path = os.path.join(self.loss_dir, "loss.png")

            if self.valid_loader is not None:
                draw_loss_curve(train_loss=self.train_loss[:epoch + 1], valid_loss=self.valid_loss[:epoch + 1], save_path=save_path)
            else:
                draw_loss_curve(train_loss=self.train_loss[:epoch + 1], save_path=save_path)

    def run_one_epoch(self, epoch):
        """
        Training
        """
        train_loss = self.run_one_epoch_train(epoch)

        if self.valid_loader is not None:
            valid_loss = self.run_one_epoch_eval(epoch)
        else:
            valid_loss = None

        return train_loss, valid_loss

    def run_one_epoch_train(self, epoch):
        # Override
        """
            Training
        """
        self.model.train()

        train_loss = 0
        n_train_batch = len(self.train_loader)

        for idx, (mixture, sources, ideal_mask, threshold_weight) in enumerate(self.train_loader):
            if self.use_cuda:
                mixture = mixture.cuda()
                sources = sources.cuda()
                ideal_mask = ideal_mask.cuda()
                threshold_weight = threshold_weight.cuda()

            mixture_amplitude = torch.abs(mixture)
            if self.target_type == "source":
                target_amplitude = torch.abs(sources)
            elif self.target_type == "oracle":
                target_amplitude = ideal_mask * mixture_amplitude
            else:
                raise NotImplementedError("Not support `target_type={}.`".format(self.target_type))

            estimated_sources_amplitude = self.model(mixture_amplitude, assignment=ideal_mask, threshold_weight=threshold_weight, n_sources=sources.size(1))

            loss = self.criterion(estimated_sources_amplitude, target_amplitude)

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
        n_sources = self.n_sources

        self.model.eval()

        valid_loss = 0
        n_valid = len(self.valid_loader.dataset)

        with torch.no_grad():
            for idx, (mixture, sources, ideal_mask, threshold_weight) in enumerate(self.valid_loader):
                """
                    mixture (batch_size, 1, n_bins, n_frames)
                    sources (batch_size, n_sources, n_bins, n_frames)
                    ideal_mask (batch_size, n_sources, n_bins, n_frames)
                    threshold_weight (batch_size, n_bins, n_frames)
                """
                if self.use_cuda:
                    mixture = mixture.cuda()
                    sources = sources.cuda()
                    threshold_weight = threshold_weight.cuda()
                    assignment = assignment.cuda()

                mixture_amplitude = torch.abs(mixture)
                if self.target_type == "source":
                    target_amplitude = torch.abs(sources)
                elif self.target_type == "oracle":
                    target_amplitude = ideal_mask * mixture_amplitude
                else:
                    raise NotImplementedError("Not support `target_type={}.`".format(self.target_type))

                estimated_sources_amplitude = self.model(mixture_amplitude, assignment=None, threshold_weight=threshold_weight, n_sources=n_sources, iter_clustering=self.iter_clustering)
                # At the test phase, assignment may be unknown.
                loss, _ = pit(self.criterion, estimated_sources_amplitude, target_amplitude, batch_mean=False)
                loss = loss.sum(dim=0)
                valid_loss += loss.item()

                if idx < 5:
                    mixture = mixture[0].cpu() # (1, n_bins, n_frames, 2)
                    mixture_amplitude = mixture_amplitude[0].cpu() # (1, n_bins, n_frames)
                    estimated_sources_amplitude = estimated_sources_amplitude[0].cpu() # (n_sources, n_bins, n_frames)

                    phase = torch.angle(mixture)
                    estimated_sources = estimated_sources_amplitude * torch.exp(1j * phase)
                    estimated_sources = istft(estimated_sources, n_fft=self.n_fft, hop_length=self.hop_length, normalized=self.normalize, window=self.window) # (n_sources, T)
                    estimated_sources = estimated_sources.cpu()

                    mixture = istft(mixture, n_fft=self.n_fft, hop_length=self.hop_length, normalized=self.normalize, window=self.window) # (1, T)
                    mixture = mixture.squeeze(dim=0) # (T,)

                    save_dir = os.path.join(self.sample_dir, "{}".format(idx + 1))
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

        valid_loss /= n_valid

        return valid_loss

    def save_model(self, epoch, model_path='./tmp.pth'):
        if isinstance(self.model, nn.DataParallel):
            config = self.model.module.get_config()
            config['state_dict'] = self.model.module.state_dict()
        else:
            config = self.model.get_config()
            config['state_dict'] = self.model.state_dict()

        config['optim_dict'], config['scheduler_dict'] = self.optimizer.state_dict(), self.scheduler.state_dict()

        config['no_improvement'] = self.no_improvement
        config['best_loss'] = self.best_loss
        config['train_loss'], config['valid_loss'] = self.train_loss, self.valid_loss

        config['epoch'] = epoch + 1

        torch.save(config, model_path)

class AdhocTester(TesterBase):
    def __init__(self, model, loader, pit_criterion, metrics, args):
        self.loader = loader

        self.model = model

        self.pit_criterion, self.metrics = pit_criterion, metrics

        self._reset(args)

    def _reset(self, args):
        # Override
        super()._reset(args)

        self.n_bins = args.n_bins
        self.n_fft, self.hop_length = args.n_fft, args.hop_length
        self.window = self.loader.dataset.window
        self.normalize = self.loader.dataset.normalize

        self.target_type = args.target_type
        self.iter_clustering = args.iter_clustering

    def run(self):
        self.model.eval()

        n_sources = self.n_sources

        test_loss = 0
        test_sdr_improvement = 0
        test_sir_improvement = 0
        test_sar = 0
        test_pesq = 0
        test_results = OrderedDict()

        n_pesq_error = 0
        n_test = len(self.loader.dataset)

        s = "ID, Loss, SDR improvement, SIR improvement, SAR, PESQ"
        for key, _ in self.metrics.items():
            s += ", {} improvement".format(key)
            test_results[key] = 0
        print(s, flush=True)

        tmp_ID = str(uuid.uuid4())
        tmp_dir = os.path.join(os.getcwd(), 'tmp-{}'.format(tmp_ID))
        os.makedirs(tmp_dir, exist_ok=True)
        shutil.copy('./PESQ', os.path.join(tmp_dir, 'PESQ'))
        os.chdir(tmp_dir)

        with torch.no_grad():
            for idx, (mixture, sources, ideal_mask, threshold_weight, T, segment_IDs) in enumerate(self.loader):
                """
                    mixture (1, 1, n_bins, n_frames)
                    sources (1, n_sources, n_bins, n_frames)
                    ideal_mask (1, n_sources, n_bins, n_frames)
                    threshold_weight (1, n_bins, n_frames)
                    T (1,)
                """
                if self.use_cuda:
                    mixture = mixture.cuda()
                    sources = sources.cuda()
                    ideal_mask = ideal_mask.cuda()
                    threshold_weight = threshold_weight.cuda()

                mixture_amplitude = torch.abs(mixture) # (1, 1, n_bins, n_frames)
                if self.target_type == "source":
                    target_amplitude = torch.abs(sources)
                elif self.target_type == "oracle":
                    target_amplitude = ideal_mask * mixture_amplitude
                else:
                    raise NotImplementedError("Not support `target_type={}.`".format(self.target_type))

                estimated_sources_amplitude = self.model(mixture_amplitude, assignment=None, threshold_weight=threshold_weight, n_sources=n_sources, iter_clustering=self.iter_clustering)
                loss, _ = self.pit_criterion(estimated_sources_amplitude, target_amplitude, batch_mean=False)
                loss = loss.sum(dim=0)

                mixture = mixture[0].cpu()
                sources = sources[0].cpu()

                mixture_amplitude = mixture_amplitude[0].cpu() # (1, n_bins, n_frames)
                estimated_sources_amplitude = estimated_sources_amplitude[0].cpu() # (n_sources, n_bins, n_frames)

                phase = torch.angle(mixture)
                estimated_sources = estimated_sources_amplitude * torch.exp(1j * phase) # (n_sources, n_bins, n_frames)

                T = T[0]  # ()
                segment_IDs = segment_IDs[0] # (n_sources,)
                mixture = istft(mixture, n_fft=self.n_fft, hop_length=self.hop_length, normalized=self.normalize, window=self.window, length=T).squeeze(dim=0) # (T,)
                sources = istft(sources, n_fft=self.n_fft, hop_length=self.hop_length, normalized=self.normalize, window=self.window, length=T) # (n_sources, T)
                estimated_sources = istft(estimated_sources, n_fft=self.n_fft, hop_length=self.hop_length, normalized=self.normalize, window=self.window, length=T) # (n_sources, T)

                repeated_mixture = torch.tile(mixture, (self.n_sources, 1))
                result_estimated = bss_eval_sources(
                    reference_sources=sources,
                    estimated_sources=estimated_sources
                )
                result_mixed = bss_eval_sources(
                    reference_sources=sources,
                    estimated_sources=repeated_mixture
                )

                sdr_improvement = torch.mean(result_estimated[0] - result_mixed[0])
                sir_improvement = torch.mean(result_estimated[1] - result_mixed[1])
                sar = torch.mean(result_estimated[2])
                perm_idx = result_estimated[3]

                norm = torch.abs(mixture).max()
                mixture /= norm
                mixture_ID = segment_IDs

                # Generate random number temporary wav file.
                random_ID = str(uuid.uuid4())

                if idx < 10 and self.out_dir is not None:
                    mixture_path = os.path.join(self.out_dir, "{}.wav".format(mixture_ID))
                    signal = mixture.unsqueeze(dim=0) if mixture.dim() == 1 else mixture
                    torchaudio.save(mixture_path, signal, sample_rate=self.sample_rate, bits_per_sample=BITS_PER_SAMPLE_WSJ0)

                for order_idx in range(self.n_sources):
                    source, estimated_source = sources[order_idx], estimated_sources[perm_idx[order_idx]]

                    # Target
                    norm = torch.abs(source).max()
                    source /= norm
                    if idx < 10 and  self.out_dir is not None:
                        source_path = os.path.join(self.out_dir, "{}_{}-target.wav".format(mixture_ID, order_idx + 1))
                        signal = source.unsqueeze(dim=0) if source.dim() == 1 else source
                        torchaudio.save(source_path, signal, sample_rate=self.sample_rate, bits_per_sample=BITS_PER_SAMPLE_WSJ0)
                    source_path = "tmp-{}-target_{}.wav".format(order_idx + 1, random_ID)
                    signal = source.unsqueeze(dim=0) if source.dim() == 1 else source
                    torchaudio.save(source_path, signal, sample_rate=self.sample_rate, bits_per_sample=BITS_PER_SAMPLE_WSJ0)

                    # Estimated source
                    norm = torch.abs(estimated_source).max()
                    estimated_source /= norm
                    if idx < 10 and  self.out_dir is not None:
                        estimated_path = os.path.join(self.out_dir, "{}_{}-estimated.wav".format(mixture_ID, order_idx + 1))
                        signal = estimated_source.unsqueeze(dim=0) if estimated_source.dim() == 1 else estimated_source
                        torchaudio.save(estimated_path, signal, sample_rate=self.sample_rate, bits_per_sample=BITS_PER_SAMPLE_WSJ0)
                    estimated_path = "tmp-{}-estimated_{}.wav".format(order_idx + 1, random_ID)
                    signal = estimated_source.unsqueeze(dim=0) if estimated_source.dim() == 1 else estimated_source
                    torchaudio.save(estimated_path, signal, sample_rate=self.sample_rate, bits_per_sample=BITS_PER_SAMPLE_WSJ0)

                pesq = 0

                for source_idx in range(self.n_sources):
                    source_path = "tmp-{}-target_{}.wav".format(source_idx + 1, random_ID)
                    estimated_path = "tmp-{}-estimated_{}.wav".format(source_idx + 1, random_ID)

                    command = "./PESQ +{} {} {}".format(self.sample_rate, source_path, estimated_path)
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

                results = self.metrics(repeated_mixture.unsqueeze(dim=0), estimated_sources[perm_idx].unsqueeze(dim=0), sources.unsqueeze(dim=0))

                s = "{}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}".format(mixture_ID, loss.item(), sdr_improvement.item(), sir_improvement.item(), sar.item(), pesq)
                for _, result in results.items():
                    s += ", {:.3f}".format(result.item())
                print(s, flush=True)

                test_loss += loss.item()
                test_sdr_improvement += sdr_improvement.item()
                test_sir_improvement += sir_improvement.item()
                test_sar += sar.item()
                test_pesq += pesq

                for key, result in results.items():
                    test_results[key] += result

        test_loss /= n_test
        test_sdr_improvement /= n_test
        test_sir_improvement /= n_test
        test_sar /= n_test
        test_pesq /= n_test

        for key in test_results.keys():
            test_results[key] /= n_test

        os.chdir("../") # back to the original directory
        shutil.rmtree(tmp_dir)

        s = "Loss: {:.3f}, SDR improvement: {:.3f}, SIR improvement: {:.3f}, SAR: {:.3f}, PESQ: {:.3f}".format(test_loss, test_sdr_improvement, test_sir_improvement, test_sar, test_pesq)

        for key, result in test_results.items():
            s += ", {} improvement: {:.3f}".format(key, result)

        print(s)
        print("Evaluation of PESQ returns error {} times".format(n_pesq_error))

class AdhocFinetuneTrainer(TrainerBase):
    def __init__(self, model, loader, criterion, optimizer, scheduler, args):
        self.train_loader, self.valid_loader = loader['train'], loader['valid']

        self.model = model

        self.criterion = criterion
        self.optimizer, self.scheduler = optimizer, scheduler

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
        self.train_loss, self.valid_loss = torch.empty(self.epochs), torch.empty(self.epochs)

        self.use_cuda = args.use_cuda

        config = torch.load(args.continue_from, map_location=lambda storage, loc: storage)
        continue_from_finetune = config.get('is_finetune') or False

        if continue_from_finetune:
            self.start_epoch = config['epoch']
            self.train_loss[:self.start_epoch] = config['train_loss'][:self.start_epoch]
            self.valid_loss[:self.start_epoch] = config['valid_loss'][:self.start_epoch]

            self.best_loss = config['best_loss']
            self.prev_loss = self.valid_loss[self.start_epoch - 1]
            self.no_improvement = config['no_improvement']

            self.optimizer.load_state_dict(config['optim_dict'])
            self.scheduler.load_state_dict(config['scheduler_dict'])
        else:
            model_path = os.path.join(self.model_dir, "best.pth")

            if os.path.exists(model_path):
                if args.overwrite:
                    print("Overwrite models.")
                else:
                    raise ValueError("{} already exists. If you continue to run, set --overwrite to be True.".format(model_path))

            self.start_epoch = 0
            self.best_loss, self.prev_loss = float('infinity'), float('infinity')
            self.no_improvement = 0

        if isinstance(self.model, nn.DataParallel):
            self.model.module.load_state_dict(config['state_dict'])
        else:
            self.model.load_state_dict(config['state_dict'])

        self.n_bins = args.n_bins
        self.n_fft, self.hop_length = args.n_fft, args.hop_length
        self.window = self.train_loader.dataset.window
        self.normalize = self.train_loader.dataset.normalize

        self.target_type = args.target_type
        self.iter_clustering = args.iter_clustering

    def run(self):
        for epoch in range(self.start_epoch, self.epochs):
            start = time.time()
            train_loss, valid_loss = self.run_one_epoch(epoch)
            end = time.time()

            s = "[Epoch {}/{}] loss (train): {:.5f}".format(epoch + 1, self.epochs, train_loss)
            self.train_loss[epoch] = train_loss

            if self.valid_loader is not None:
                s += ", loss (valid): {:.5f}".format(valid_loss)
                self.valid_loss[epoch] = valid_loss

            s += ", {:.3f} [sec]".format(end - start)
            print(s, flush=True)

            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(valid_loss)
            else:
                self.scheduler.step()

            if self.valid_loader is not None:
                if valid_loss < self.best_loss:
                    self.best_loss = valid_loss
                    self.no_improvement = 0
                    model_path = os.path.join(self.model_dir, "best.pth")
                    self.save_model(epoch, model_path)
                else:
                    if valid_loss >= self.prev_loss:
                        self.no_improvement += 1
                        if self.no_improvement >= NO_IMPROVEMENT:
                            print("Stop training")
                            break
                    else:
                        self.no_improvement = 0

                self.prev_loss = valid_loss

            model_path = os.path.join(self.model_dir, "last.pth")
            self.save_model(epoch, model_path)

            save_path = os.path.join(self.loss_dir, "loss.png")

            if self.valid_loader is not None:
                draw_loss_curve(train_loss=self.train_loss[:epoch + 1], valid_loss=self.valid_loss[:epoch + 1], save_path=save_path)
            else:
                draw_loss_curve(train_loss=self.train_loss[:epoch + 1], save_path=save_path)

    def run_one_epoch(self, epoch):
        train_loss = self.run_one_epoch_train(epoch)

        if self.valid_loader is not None:
            valid_loss = self.run_one_epoch_eval(epoch)
        else:
            valid_loss = None

        return train_loss, valid_loss

    def run_one_epoch_train(self, epoch):
        # Override
        """
            Training
        """
        self.model.train()

        train_loss = 0
        n_train_batch = len(self.train_loader)

        for idx, (mixture, sources, ideal_mask, threshold_weight) in enumerate(self.train_loader):
            if self.use_cuda:
                mixture, sources = mixture.cuda(), sources.cuda()
                ideal_mask = ideal_mask.cuda()
                threshold_weight = threshold_weight.cuda()

            mixture_amplitude = torch.abs(mixture)
            if self.target_type == "source":
                target_amplitude = torch.abs(sources)
            elif self.target_type == "oracle":
                target_amplitude = ideal_mask * mixture_amplitude
            else:
                raise NotImplementedError("Not support `target_type={}.`".format(self.target_type))

            estimated_sources_amplitude = self.model(mixture_amplitude, assignment=ideal_mask, threshold_weight=threshold_weight, n_sources=sources.size(1))

            loss = self.criterion(estimated_sources_amplitude, target_amplitude)

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
        n_sources = self.n_sources

        self.model.eval()

        valid_loss = 0
        n_valid = len(self.valid_loader.dataset)

        with torch.no_grad():
            for idx, (mixture, sources, ideal_mask, threshold_weight) in enumerate(self.valid_loader):
                """
                    mixture (batch_size, 1, n_bins, n_frames)
                    sources (batch_size, n_sources, n_bins, n_frames)
                    ideal_mask (batch_size, n_sources, n_bins, n_frames)
                    threshold_weight (batch_size, n_bins, n_frames)
                """
                if self.use_cuda:
                    mixture = mixture.cuda()
                    sources = sources.cuda()
                    threshold_weight = threshold_weight.cuda()
                    assignment = assignment.cuda()

                mixture_amplitude = torch.abs(mixture)
                if self.target_type == "source":
                    target_amplitude = torch.abs(sources)
                elif self.target_type == "oracle":
                    target_amplitude = ideal_mask * mixture_amplitude
                else:
                    raise NotImplementedError("Not support `target_type={}.`".format(self.target_type))

                estimated_sources_amplitude = self.model(mixture_amplitude, assignment=None, threshold_weight=threshold_weight, n_sources=n_sources, iter_clustering=self.iter_clustering)
                # At the test phase, assignment may be unknown.
                loss, _ = pit(self.criterion, estimated_sources_amplitude, target_amplitude, batch_mean=False)
                loss = loss.sum(dim=0)
                valid_loss += loss.item()

                if idx < 5:
                    mixture = mixture[0].cpu() # (1, n_bins, n_frames, 2)
                    mixture_amplitude = mixture_amplitude[0].cpu() # (1, n_bins, n_frames)
                    estimated_sources_amplitude = estimated_sources_amplitude[0].cpu() # (n_sources, n_bins, n_frames)

                    phase = torch.angle(mixture)
                    estimated_sources = estimated_sources_amplitude * torch.exp(1j * phase)
                    estimated_sources = istft(estimated_sources, n_fft=self.n_fft, hop_length=self.hop_length, normalized=self.normalize, window=self.window) # (n_sources, T)
                    estimated_sources = estimated_sources.cpu()

                    mixture = istft(mixture, n_fft=self.n_fft, hop_length=self.hop_length, normalized=self.normalize, window=self.window) # (1, T)
                    mixture = mixture.squeeze(dim=0) # (T,)

                    save_dir = os.path.join(self.sample_dir, "{}".format(idx + 1))
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

        valid_loss /= n_valid

        return valid_loss

    def save_model(self, epoch, model_path='./tmp.pth'):
        if isinstance(self.model, nn.DataParallel):
            config = self.model.module.get_config()
            config['state_dict'] = self.model.module.state_dict()
        else:
            config = self.model.get_config()
            config['state_dict'] = self.model.state_dict()

        config['optim_dict'], config['scheduler_dict'] = self.optimizer.state_dict(), self.scheduler.state_dict()

        config['no_improvement'] = self.no_improvement
        config['best_loss'] = self.best_loss
        config['train_loss'], config['valid_loss'] = self.train_loss, self.valid_loss

        config['epoch'] = epoch + 1
        config['is_finetune'] = True # For finetuner

        torch.save(config, model_path)

class FixedAttractorComputer:
    def __init__(self, model, loader, args):
        self.loader = loader

        self.model = model

        self._reset(args)

    def _reset(self, args):
        self.use_cuda = args.use_cuda

        self.base_model_path, self.wrapper_model_dir = args.base_model_path, args.wrapper_model_dir

        config = torch.load(self.base_model_path, map_location=lambda storage, loc: storage)

        if isinstance(self.model, nn.DataParallel):
            self.model.module.load_state_dict(config['state_dict'])
        else:
            self.model.load_state_dict(config['state_dict'])

        os.makedirs(self.wrapper_model_dir, exist_ok=True)

        self.wrapper_class = args.wrapper_class
        self.iter_clustering = args.iter_clustering

    def run(self):
        self.model.eval()

        attractors = []

        with torch.no_grad():
            for mixture, sources, assignment, threshold_weight in self.loader:
                if self.use_cuda:
                    mixture = mixture.cuda()
                    assignment = assignment.cuda()
                    threshold_weight = threshold_weight.cuda()

                n_sources = sources.size(1)
                mixture_amplitude = torch.abs(mixture)
                _, _, attractor = self.model.extract_latent(mixture_amplitude, assignment=assignment, threshold_weight=threshold_weight, n_sources=n_sources, iter_clustering=self.iter_clustering)
                attractors.append(attractor)

        attractors = torch.cat(attractors, dim=0) # (len(dataset), n_sources, embed_dim)
        attractors = attractors.view(-1, attractors.size(-1)) # (len(dataset) * n_sources, embed_dim)

        kmeans = KMeans(K=n_sources)
        _ = kmeans(attractors) # (n_sources, embed_dim)

        base_model_filename = os.path.basename(self.base_model_path)
        wrapper_model_path = os.path.join(self.wrapper_model_dir, base_model_filename)
        self.save_model(self.wrapper_class, kmeans.centroids, model_path=wrapper_model_path)

    def save_model(self, wrapper_class, attractor, model_path="./tmp.pth"):
        if isinstance(self.model, nn.DataParallel):
            base_model = self.model.module
        else:
            base_model = self.model

        wrapper_model = wrapper_class(base_model, attractor)

        config = base_model.get_config()
        config["state_dict"] = wrapper_model.state_dict()
        config["attractor_size"] = attractor.size()

        torch.save(config, model_path)

class FixedAttractorTester(TesterBase):
    def __init__(self, model, loader, pit_criterion, metrics, args):
        self.loader = loader

        self.model = model

        self.pit_criterion, self.metrics = pit_criterion, metrics

        self._reset(args)

    def _reset(self, args):
        # Override
        super()._reset(args)

        self.n_bins = args.n_bins
        self.n_fft, self.hop_length = args.n_fft, args.hop_length
        self.window = self.loader.dataset.window
        self.normalize = self.loader.dataset.normalize

        self.target_type = args.target_type

    def run(self):
        self.model.eval()

        test_loss = 0
        test_sdr_improvement = 0
        test_sir_improvement = 0
        test_sar = 0
        test_pesq = 0
        test_results = OrderedDict()

        n_pesq_error = 0
        n_test = len(self.loader.dataset)

        s = "ID, Loss, SDR improvement, SIR improvement, SAR, PESQ"
        for key, _ in self.metrics.items():
            s += ", {} improvement".format(key)
            test_results[key] = 0
        print(s, flush=True)

        tmp_ID = str(uuid.uuid4())
        tmp_dir = os.path.join(os.getcwd(), 'tmp-{}'.format(tmp_ID))
        os.makedirs(tmp_dir, exist_ok=True)
        shutil.copy('./PESQ', os.path.join(tmp_dir, 'PESQ'))
        os.chdir(tmp_dir)

        with torch.no_grad():
            for idx, (mixture, sources, ideal_mask, _, T, segment_IDs) in enumerate(self.loader):
                """
                    mixture (1, 1, n_bins, n_frames)
                    sources (1, n_sources, n_bins, n_frames)
                    ideal_mask (1, n_sources, n_bins, n_frames)
                    threshold_weight (1, n_bins, n_frames)
                    T (1,)
                """
                if self.use_cuda:
                    mixture = mixture.cuda()
                    sources = sources.cuda()

                mixture_amplitude = torch.abs(mixture) # (1, 1, n_bins, n_frames)
                if self.target_type == "source":
                    target_amplitude = torch.abs(sources)
                elif self.target_type == "oracle":
                    target_amplitude = ideal_mask * mixture_amplitude
                else:
                    raise NotImplementedError("Not support `target_type={}.`".format(self.target_type))

                estimated_sources_amplitude = self.model(mixture_amplitude)
                loss, _ = self.pit_criterion(estimated_sources_amplitude, target_amplitude, batch_mean=False)
                loss = loss.sum(dim=0)

                mixture = mixture[0].cpu()
                sources = sources[0].cpu()

                mixture_amplitude = mixture_amplitude[0].cpu() # (1, n_bins, n_frames)
                estimated_sources_amplitude = estimated_sources_amplitude[0].cpu() # (n_sources, n_bins, n_frames)

                phase = torch.angle(mixture)
                estimated_sources = estimated_sources_amplitude * torch.exp(1j * phase) # (n_sources, n_bins, n_frames)

                T = T[0]  # ()
                segment_IDs = segment_IDs[0] # (n_sources,)
                mixture = istft(mixture, n_fft=self.n_fft, hop_length=self.hop_length, normalized=self.normalize, window=self.window, length=T).squeeze(dim=0) # (T,)
                sources = istft(sources, n_fft=self.n_fft, hop_length=self.hop_length, normalized=self.normalize, window=self.window, length=T) # (n_sources, T)
                estimated_sources = istft(estimated_sources, n_fft=self.n_fft, hop_length=self.hop_length, normalized=self.normalize, window=self.window, length=T) # (n_sources, T)

                repeated_mixture = torch.tile(mixture, (self.n_sources, 1))
                result_estimated = bss_eval_sources(
                    reference_sources=sources,
                    estimated_sources=estimated_sources
                )
                result_mixed = bss_eval_sources(
                    reference_sources=sources,
                    estimated_sources=repeated_mixture
                )

                sdr_improvement = torch.mean(result_estimated[0] - result_mixed[0])
                sir_improvement = torch.mean(result_estimated[1] - result_mixed[1])
                sar = torch.mean(result_estimated[2])
                perm_idx = result_estimated[3]

                norm = torch.abs(mixture).max()
                mixture /= norm
                mixture_ID = segment_IDs

                # Generate random number temporary wav file.
                random_ID = str(uuid.uuid4())

                if idx < 10 and self.out_dir is not None:
                    mixture_path = os.path.join(self.out_dir, "{}.wav".format(mixture_ID))
                    signal = mixture.unsqueeze(dim=0) if mixture.dim() == 1 else mixture
                    torchaudio.save(mixture_path, signal, sample_rate=self.sample_rate, bits_per_sample=BITS_PER_SAMPLE_WSJ0)

                for order_idx in range(self.n_sources):
                    source, estimated_source = sources[order_idx], estimated_sources[perm_idx[order_idx]]

                    # Target
                    norm = torch.abs(source).max()
                    source /= norm
                    if idx < 10 and  self.out_dir is not None:
                        source_path = os.path.join(self.out_dir, "{}_{}-target.wav".format(mixture_ID, order_idx + 1))
                        signal = source.unsqueeze(dim=0) if source.dim() == 1 else source
                        torchaudio.save(source_path, signal, sample_rate=self.sample_rate, bits_per_sample=BITS_PER_SAMPLE_WSJ0)
                    source_path = "tmp-{}-target_{}.wav".format(order_idx + 1, random_ID)
                    signal = source.unsqueeze(dim=0) if source.dim() == 1 else source
                    torchaudio.save(source_path, signal, sample_rate=self.sample_rate, bits_per_sample=BITS_PER_SAMPLE_WSJ0)

                    # Estimated source
                    norm = torch.abs(estimated_source).max()
                    estimated_source /= norm
                    if idx < 10 and  self.out_dir is not None:
                        estimated_path = os.path.join(self.out_dir, "{}_{}-estimated.wav".format(mixture_ID, order_idx + 1))
                        signal = estimated_source.unsqueeze(dim=0) if estimated_source.dim() == 1 else estimated_source
                        torchaudio.save(estimated_path, signal, sample_rate=self.sample_rate, bits_per_sample=BITS_PER_SAMPLE_WSJ0)
                    estimated_path = "tmp-{}-estimated_{}.wav".format(order_idx + 1, random_ID)
                    signal = estimated_source.unsqueeze(dim=0) if estimated_source.dim() == 1 else estimated_source
                    torchaudio.save(estimated_path, signal, sample_rate=self.sample_rate, bits_per_sample=BITS_PER_SAMPLE_WSJ0)

                pesq = 0

                for source_idx in range(self.n_sources):
                    source_path = "tmp-{}-target_{}.wav".format(source_idx + 1, random_ID)
                    estimated_path = "tmp-{}-estimated_{}.wav".format(source_idx + 1, random_ID)

                    command = "./PESQ +{} {} {}".format(self.sample_rate, source_path, estimated_path)
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

                results = self.metrics(repeated_mixture.unsqueeze(dim=0), estimated_sources[perm_idx].unsqueeze(dim=0), sources.unsqueeze(dim=0))

                s = "{}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}".format(mixture_ID, loss.item(), sdr_improvement.item(), sir_improvement.item(), sar.item(), pesq)
                for _, result in results.items():
                    s += ", {:.3f}".format(result.item())
                print(s, flush=True)

                test_loss += loss.item()
                test_sdr_improvement += sdr_improvement.item()
                test_sir_improvement += sir_improvement.item()
                test_sar += sar.item()
                test_pesq += pesq

                for key, result in results.items():
                    test_results[key] += result

        test_loss /= n_test
        test_sdr_improvement /= n_test
        test_sir_improvement /= n_test
        test_sar /= n_test
        test_pesq /= n_test

        for key in test_results.keys():
            test_results[key] /= n_test

        os.chdir("../") # back to the original directory
        shutil.rmtree(tmp_dir)

        s = "Loss: {:.3f}, SDR improvement: {:3f}, SIR improvement: {:3f}, SAR: {:3f}, PESQ: {:.3f}".format(test_loss, test_sdr_improvement, test_sir_improvement, test_sar, test_pesq)

        for key, result in test_results.items():
            s += ", {} improvement: {:.3f}".format(key, result)

        print(s)
        print("Evaluation of PESQ returns error {} times".format(n_pesq_error))