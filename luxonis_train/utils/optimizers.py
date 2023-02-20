from torch.optim import *

def get_optimizer(model_params, name, **kwargs):
    return eval(name)(params=model_params, **kwargs)