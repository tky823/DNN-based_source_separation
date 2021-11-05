from augmentation import RandomFlip, RandomGain

class SequentialAugmentation:
    def __init__(self, *args):
        super().__init__()

        self.processes = list(args)

    def __call__(self, input):
        x = input
        for process in self.processes:
            x = process(x)
        output = x

        return output
    
    def append(self, __object):
        self.processes.append(__object)

def choose_augmentation(name, **kwargs):
    if name == 'random_flip':
        return RandomFlip(**kwargs)
    elif name == 'random_scaling':
        return RandomGain(**kwargs)
    elif name == 'random_gain':
        return RandomGain(**kwargs)
    else:
        raise NotImplementedError("Not support {}.".format(name))
