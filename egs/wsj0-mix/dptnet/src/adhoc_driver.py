import os
import time

import torch
import torch.nn as nn

from utils.utils import draw_loss_curve
from driver import TrainerBase, TesterBase

class AdhocTrainer(TrainerBase):
    def __init__(self, model, loader, pit_criterion, optimizer, args):
        super().__init__(model, loader, pit_criterion, optimizer, args)

    def _reset(self, args):
        super()._reset(args)

        self.d_model = args.sep_bottleneck_channels

        # Learning rate
        self.k1, self.k2 = args.k1, args.k2
        self.warmup_steps = args.warmup_steps

        if args.continue_from:
            config = torch.load(args.continue_from, map_location=lambda storage, loc: storage)
            self.step = config['step']
        else:
            self.step = 0
    
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
                else:
                    self.no_improvement = 0
            
            self.prev_loss = valid_loss
            
            model_path = os.path.join(self.model_dir, "last.pth")
            self.save_model(epoch, model_path)
            
            save_path = os.path.join(self.loss_dir, "loss.png")
            draw_loss_curve(train_loss=self.train_loss[:epoch + 1], valid_loss=self.valid_loss[:epoch + 1], save_path=save_path)

            if self.no_improvement >= 10:
                print("Stop training.")
                break
    
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
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
            
            self.update_lr(epoch)
            self.optimizer.step()
            
            train_loss += loss.item()
            
            if (idx + 1)%100 == 0:
                print("[Epoch {}/{}] iter {}/{} loss: {:.5f}".format(epoch + 1, self.epochs, idx + 1, n_train_batch, loss.item()), flush=True)
        
        train_loss /= n_train_batch
        
        return train_loss

    def update_lr(self, epoch):
        """
        Update learning rate for CURRENT step
        """
        step = self.step
        warmup_steps = self.warmup_steps

        if step > warmup_steps:
            k = self.k2
            lr = k * 0.98 ** ((epoch + 1) // 2)
        else:
            k = self.k1
            d_model = self.d_model
            lr = k * d_model ** (-0.5) * (step + 1) * warmup_steps ** (-1.5)

        prev_lr = None

        for param_group in self.optimizer.param_groups:
            if (step + 1) % 100 == 0 and prev_lr is None:
                prev_lr = param_group['lr']
                if lr == prev_lr:
                    break
                else:
                    print("Learning rate: {} -> {}".format(prev_lr, lr))
            param_group['lr'] = lr

        self.step = step + 1
    
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
        config['step'] = self.step # self.step is already updated in `update_lr`, so you don't have to plus 1.
        
        torch.save(config, model_path)

class Tester(TesterBase):
    def __init__(self, model, loader, pit_criterion, args):
        super().__init__(model, loader, pit_criterion, args)