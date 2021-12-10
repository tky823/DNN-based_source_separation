import os

import torch
import torchvision
import torch.nn as nn

from utils.utils import draw_loss_curve
from criterion.entropy import BinaryCrossEntropy
from adhoc_criterion import KLdivergence

EPS = 1e-12

class Trainer:
    def __init__(self, model, loader, optimizer, args):
        self.loader = loader
        
        self.model = model
        self.optimizer = optimizer
        
        self._reset(args)
        
    def _reset(self, args):
        self.num_samples = args.num_samples
        self.latent_dim = args.latent_dim

        self.max_norm = args.max_norm
        
        self.model_dir = args.model_dir
        self.loss_dir = args.loss_dir
        self.sample_dir = args.sample_dir
        
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.loss_dir, exist_ok=True)
        os.makedirs(self.sample_dir, exist_ok=True)
        
        self.epochs = args.epochs
        self.use_cuda = args.use_cuda

        self.kl_divergence = KLdivergence()
        self.reconstruction = BinaryCrossEntropy(reduction="sum")
        
        # Loss
        self.criterions = ["loss", "kl", "reconstruction"]

        self.train_loss = {
            key: torch.empty(self.epochs) for key in self.criterions
        }
        self.valid_loss = {
            key: torch.empty(self.epochs) for key in self.criterions
        }

        self.best_loss, self.prev_loss = {}, {}
        
        if args.continue_from:
            config = torch.load(args.continue_from, map_location=lambda storage, loc: storage)
            
            self.start_epoch = config['epoch']
            
            for key in self.criterions:
                self.train_loss[key][:self.start_epoch] = config['train_loss'][key][:self.start_epoch]
                self.valid_loss[key][:self.start_epoch] = config['valid_loss'][key][:self.start_epoch]
                self.best_loss[key] = config['best_loss'][key]
                self.prev_loss[key] = self.valid_loss[key][self.start_epoch - 1]
            
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
            
            for key in self.criterions:
                self.best_loss[key] = float('infinity')
                self.prev_loss[key] = float('infinity')
            
            self.no_improvement = 0
    
    def save_model(self, save_path, epoch=0):
        config = {}

        config['epoch'] = epoch + 1

        config['state_dict'] = self.model.state_dict()
        config['optim_dict'] = self.optimizer.state_dict()

        config['train_loss'] = self.train_loss
        config['valid_loss'] = self.valid_loss
        config['best_loss'] = self.best_loss

        config['no_improvement'] = self.no_improvement
        
        torch.save(config, save_path)
        
    def run(self):
        for epoch in range(self.epochs):
            train_loss, valid_loss = self.run_one_epoch(epoch)

            s = "[Epoch {}/{}]".format(epoch + 1, self.epochs)
            s += "Train Lower Bound:{:.5e}, (".format(train_loss["loss"])

            for key in self.criterions:
                if key == "loss":
                    continue
                s += "{}: {:.5e}, ".format(key, train_loss[key])
            
            s = s[:-2]
            s += "), Valid Lower Bound:{:.5e}, (".format(valid_loss["loss"])

            for key in self.criterions:
                if key == "loss":
                    continue
                s += "{}: {:.5e}, ".format(key, valid_loss[key])

            s = s[:-2]
            s += ")"
            print(s, flush=True)

            for key, item in train_loss.items():
                self.train_loss[key][epoch] = item

            for key, item in valid_loss.items():
                self.valid_loss[key][epoch] = item
            
            if valid_loss["loss"] < self.best_loss["loss"]:
                self.best_loss = valid_loss
                self.no_improvement = 0
                model_path = os.path.join(self.model_dir, "best.pth")
                self.save_model(model_path, epoch=epoch)
            else:
                self.no_improvement += 1
                if self.no_improvement % 5 == 0:
                    for param_group in self.optimizer.param_groups:
                        prev_lr = param_group['lr']
                        lr = 0.5 * prev_lr
                        print("Learning rate: {} -> {}".format(prev_lr, lr))
                        param_group['lr'] = lr

            self.prev_loss = valid_loss
            
            model_path = os.path.join(self.model_dir, "last.pth")
            self.save_model(model_path, epoch=epoch)
            
            for key in self.criterions:
                save_path = os.path.join(self.loss_dir, "{}.png".format(key))
                draw_loss_curve(train_loss=self.train_loss[key][:epoch + 1], valid_loss=self.valid_loss[key][:epoch + 1], save_path=save_path)
    
    def run_one_epoch_train(self, epoch):
        n_train_batch = len(self.loader['train'])
        train_loss = 0
        train_kl_loss = 0
        train_reconstruction_loss = 0
    
        self.model.train()
    
        for input, _ in self.loader['train']:
            if self.use_cuda:
                input = input.cuda()

            output, _, mean, var = self.model(input, num_samples=self.num_samples, return_params=True)

            kl_loss = self.kl_divergence(mean, var)
            reconstruction_loss = self.reconstruction(output, input.unsqueeze(dim=1))
            loss = kl_loss + reconstruction_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            train_loss += loss.item()
            train_kl_loss += kl_loss.item()
            train_reconstruction_loss += reconstruction_loss.item()
            
        train_loss /= n_train_batch
        train_kl_loss /= n_train_batch
        train_reconstruction_loss /= n_train_batch

        train_loss = {
            "loss": train_loss,
            "kl": train_kl_loss,
            "reconstruction": train_reconstruction_loss
        }

        return train_loss

    def run_one_epoch_eval(self, epoch):
        n_valid = len(self.loader['valid'].dataset)
        valid_loss = 0
        valid_kl_loss = 0
        valid_reconstruction_loss = 0
        
        self.model.eval()
        
        with torch.no_grad():
            for idx, (input, _) in enumerate(self.loader['valid']):
                if self.use_cuda:
                    input = input.cuda()

                output, _, mean, var = self.model(input, return_params=True)
                
                kl_loss = self.kl_divergence(mean, var, batch_mean=False)
                reconstruction_loss = self.reconstruction(output, input.unsqueeze(dim=1), batch_mean=False)
                loss = kl_loss + reconstruction_loss
                
                valid_loss += loss.sum().item()
                valid_kl_loss += kl_loss.sum().item()
                valid_reconstruction_loss += reconstruction_loss.sum().item()

                if idx < 5:
                    input = input[0].cpu().view(1, 28, 28)
                    output = output[0, 0].cpu().view(1, 28, 28)
                    input = torchvision.transforms.functional.to_pil_image(input)
                    output = torchvision.transforms.functional.to_pil_image(output)
                    
                    save_dir = os.path.join(self.sample_dir, "{}".format(idx + 1))
                    os.makedirs(save_dir, exist_ok=True)

                    save_path = os.path.join(save_dir, "input.png")
                    input.save(save_path)

                    save_path = os.path.join(save_dir, "{}.png".format(epoch + 1))
                    output.save(save_path)
            
            # Random image
            latent_dim = self.latent_dim
            latent = torch.randn((self.loader["valid"].batch_size, 1, latent_dim)) # (batch_size, num_samples, latent_dim), where num_samples = 1.

            if self.use_cuda:
                latent = latent.cuda()

            output = self.model.decoder(latent) # (batch_size, num_samples, in_channels)

            for idx, _output in enumerate(output):
                _output = _output.cpu().view(1, 28, 28)
                _output = torchvision.transforms.functional.to_pil_image(_output)

                save_dir = os.path.join(self.sample_dir, "random-{}".format(idx + 1))
                os.makedirs(save_dir, exist_ok=True)

                save_path = os.path.join(save_dir, "{}.png".format(epoch + 1))
                _output.save(save_path)

        valid_loss /= n_valid
        valid_kl_loss /= n_valid
        valid_reconstruction_loss /= n_valid

        valid_loss = {
            "loss": valid_loss,
            "kl": valid_kl_loss,
            "reconstruction": valid_reconstruction_loss
        }
        
        return valid_loss

    def run_one_epoch(self, epoch):
        train_loss = self.run_one_epoch_train(epoch)
        valid_loss = self.run_one_epoch_eval(epoch)

        return train_loss, valid_loss
