from torch.optim.lr_scheduler import *

def init_scheduler(optimizer, name, **kwargs):
    """ Initializes and returns scheduler based on provided name and config"""
    return eval(name)(optimizer=optimizer, **kwargs)