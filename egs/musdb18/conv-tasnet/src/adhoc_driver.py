from driver import TrainerBase, TesterBase

class AdhocTrainer(TrainerBase):
    def __init__(self, model, loader, criterion, optimizer, args):
        super().__init__(model, loader, criterion, optimizer, args)

class Tester(TesterBase):
    def __init__(self, model, loader, criterion, args):
        super().__init__(model, loader, criterion, args)