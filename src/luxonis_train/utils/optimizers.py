from torch import optim
from luxonis_train.utils.registry import OPTIMIZERS

OPTIMIZERS.register_module(module=optim.Adadelta)
OPTIMIZERS.register_module(module=optim.Adagrad)
OPTIMIZERS.register_module(module=optim.Adam)
OPTIMIZERS.register_module(module=optim.AdamW)
OPTIMIZERS.register_module(module=optim.SparseAdam)
OPTIMIZERS.register_module(module=optim.Adamax)
OPTIMIZERS.register_module(module=optim.ASGD)
OPTIMIZERS.register_module(module=optim.LBFGS)
OPTIMIZERS.register_module(module=optim.NAdam)
OPTIMIZERS.register_module(module=optim.RAdam)
OPTIMIZERS.register_module(module=optim.RMSprop)
OPTIMIZERS.register_module(module=optim.SGD)
