import os
import time

from utils.utils import draw_loss_curve
from driver import TrainerBase, TesterBase

class AdhocTrainer(TrainerBase):
    def __init__(self, model, loader, pit_criterion, optimizer, args):
        super().__init__(model, loader, pit_criterion, optimizer, args)
    
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
                self.no_improvement = 0
            
            self.prev_loss = valid_loss
            
            model_path = os.path.join(self.model_dir, "last.pth")
            self.save_model(epoch, model_path)
            
            save_path = os.path.join(self.loss_dir, "loss.png")
            draw_loss_curve(train_loss=self.train_loss[:epoch + 1], valid_loss=self.valid_loss[:epoch + 1], save_path=save_path)

class Tester(TesterBase):
    def __init__(self, model, loader, pit_criterion, args):
        super().__init__(model, loader, pit_criterion, args)