from torch.optim import *

def init_optimizer(model_params, name, **kwargs):
    """ Initializes and returns optimizer based on provided name and config"""
    return eval(name)(params=model_params, **kwargs)