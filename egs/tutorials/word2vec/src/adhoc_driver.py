import os
import time

import matplotlib.pyplot as plt

import torch

class Trainer:
    def __init__(self, model, loader, criterion, optimizer, config):
        self.train_loader, self.valid_loader = loader["train"], loader["valid"]
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

        self._reset(config)

    def _reset(self, config):
        self.context_size = config["context_size"]
        self.epochs = config["epochs"]

        self.train_loss = torch.empty(self.epochs)
        self.valid_loss = torch.empty(self.epochs)

        if config["continue_from"]:
            package = torch.load(config["continue_from"], map_location=lambda storage, loc: storage)

            self.start_epoch = package["epoch"]
            self.model.load_state_dict(package["state_dict"])
            self.optimizer.load_state_dict(package["optim_dict"])

            self.train_loss[: self.start_epoch] = package["train_loss"][: self.start_epoch]
            self.valid_loss[: self.start_epoch] = package["valid_loss"][: self.start_epoch]
            self.best_loss = package["best_loss"]
        else:
            self.start_epoch = 0
            self.best_loss = float("inf")

        self.model_dir = config["model_dir"]
        self.loss_dir = config["loss_dir"]

        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.loss_dir, exist_ok=True)

        self.use_cuda = config["system"]["use_cuda"]

    def run(self):
        for epoch in range(self.start_epoch, self.epochs):
            start = time.time()
            train_loss, valid_loss = self.run_one_epoch(epoch)
            end = time.time()

            print("[Epoch {}/{}] (train) {:.3f}, (valid) {:.3f}, {:.3f}[sec]".format(epoch + 1, self.epochs, train_loss, valid_loss, end - start))

    def run_one_epoch(self, epoch):
        train_loss = self.run_one_epoch_train(epoch)
        valid_loss = self.run_one_epoch_eval(epoch)

        self.train_loss[epoch] = train_loss
        self.valid_loss[epoch] = valid_loss

        if valid_loss < self.best_loss:
            self.best_loss = valid_loss
            model_path = os.path.join(self.model_dir, "best.pth")
            self.save_model(epoch, model_path)

        model_path = os.path.join(self.model_dir, "last.pth")
        self.save_model(epoch, model_path)

        save_path = os.path.join(self.loss_dir, "loss.png")
        self.draw_loss_curve(self.train_loss[:epoch + 1], self.valid_loss[:epoch + 1], save_path=save_path)

        return train_loss, valid_loss

    def run_one_epoch_train(self, epoch):
        raise NotImplementedError("Implement 'run_one_epoch_train'")

    def run_one_epoch_eval(self, epoch):
        raise NotImplementedError("Implement 'run_one_epoch_train'")

    def save_model(self, epoch, model_path):
        config = {}
        config["state_dict"] = self.model.state_dict()
        config["optim_dict"] = self.optimizer.state_dict()

        config["epoch"] = epoch + 1
        config["epochs"] = self.epochs

        config["train_loss"] = self.train_loss
        config["valid_loss"] = self.valid_loss
        config["best_loss"] = self.best_loss

        torch.save(config, model_path)

    def draw_loss_curve(self, train_loss, valid_loss=None, save_path="loss.png"):
        epochs = list(range(1, len(train_loss) + 1))

        plt.figure()
        plt.plot(epochs, train_loss)
        plt.plot(epochs, valid_loss)
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()