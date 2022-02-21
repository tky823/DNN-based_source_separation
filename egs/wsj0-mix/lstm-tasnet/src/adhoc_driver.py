import os
import time

import torch
import torch.nn as nn

from utils.utils import draw_loss_curve
from driver import TrainerBase, TesterBase

HALVE_LR = 3

class AdhocTrainer(TrainerBase):
    def __init__(self, model, loader, pit_criterion, optimizer, args):
        super().__init__(model, loader, pit_criterion, optimizer, args)

    def run(self):
        for epoch in range(self.start_epoch, self.epochs):
            start = time.time()
            train_loss, valid_loss = self.run_one_epoch(epoch)
            end = time.time()

            print("[Epoch {}/{}] loss (train): {:.5f}, loss (valid): {:.5f}, {:.3f} [sec]".format(epoch + 1, self.epochs, train_loss, valid_loss, end - start), flush=True)

            self.train_loss[epoch], self.valid_loss[epoch] = train_loss, valid_loss

            if valid_loss < self.best_loss:
                self.best_loss = valid_loss
                self.no_improvement = 0
                model_path = os.path.join(self.model_dir, "best.pth")
                self.save_model(epoch, model_path)
            else:
                if valid_loss >= self.prev_loss:
                    self.no_improvement += 1
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

class FinetuneTrainer(TrainerBase):
    def __init__(self, model, loader, pit_criterion, optimizer, args):
        super().__init__(model, loader, pit_criterion, optimizer, args)

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

        # Continue from
        config = torch.load(args.continue_from, map_location=lambda storage, loc: storage)

        continue_from_finetune = config.get('is_finetune') or False

        if continue_from_finetune:
            self.start_epoch = config['epoch']
            self.train_loss[:self.start_epoch] = config['train_loss'][:self.start_epoch]
            self.valid_loss[:self.start_epoch] = config['valid_loss'][:self.start_epoch]
            self.best_loss = config['best_loss']
            self.prev_loss = self.valid_loss[self.start_epoch - 1]
            self.no_improvement = config['no_improvement']
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

        self.optimizer.load_state_dict(config['optim_dict'])

    def run(self):
        for epoch in range(self.start_epoch, self.epochs):
            start = time.time()
            train_loss, valid_loss = self.run_one_epoch(epoch)
            end = time.time()

            print("[Epoch {}/{}] loss (train): {:.5f}, loss (valid): {:.5f}, {:.3f} [sec]".format(epoch + 1, self.epochs, train_loss, valid_loss, end - start), flush=True)

            self.train_loss[epoch], self.valid_loss[epoch] = train_loss, valid_loss

            if valid_loss < self.best_loss:
                self.best_loss = valid_loss
                self.no_improvement = 0
                model_path = os.path.join(self.model_dir, "best.pth")
                self.save_model(epoch, model_path)
            else:
                if valid_loss >= self.prev_loss:
                    self.no_improvement += 1
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

    def save_model(self, epoch, model_path='./tmp.pth'):
        if isinstance(self.model, nn.DataParallel):
            config = self.model.module.get_config()
            config['state_dict'] = self.model.module.state_dict()
        else:
            config = self.model.get_config()
            config['state_dict'] = self.model.state_dict()

        config['optim_dict'] = self.optimizer.state_dict()

        config['no_improvement'] = self.no_improvement
        config['best_loss'] = self.best_loss
        config['train_loss'], config['valid_loss'] = self.train_loss, self.valid_loss

        config['epoch'] = epoch + 1
        config['is_finetune'] = True # For finetuner

        torch.save(config, model_path)

class AdhocFinetuneTrainer(FinetuneTrainer):
    def __init__(self, model, loader, pit_criterion, optimizer, args):
        super().__init__(model, loader, pit_criterion, optimizer, args)

class Tester(TesterBase):
    def __init__(self, model, loader, pit_criterion, args):
        super().__init__(model, loader, pit_criterion, args)