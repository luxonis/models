from torch.optim.lr_scheduler import *

def get_scheduler(optimizer, name, **kwargs):
    return eval(name)(optimizer=optimizer, **kwargs)