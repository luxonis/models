from torch.optim import lr_scheduler
from luxonis_train.utils.registry import SCHEDULERS

SCHEDULERS.register_module(module=lr_scheduler.LambdaLR)
SCHEDULERS.register_module(module=lr_scheduler.MultiplicativeLR)
SCHEDULERS.register_module(module=lr_scheduler.StepLR)
SCHEDULERS.register_module(module=lr_scheduler.MultiStepLR)
SCHEDULERS.register_module(module=lr_scheduler.ConstantLR)
SCHEDULERS.register_module(module=lr_scheduler.LinearLR)
SCHEDULERS.register_module(module=lr_scheduler.ExponentialLR)
SCHEDULERS.register_module(module=lr_scheduler.PolynomialLR)
SCHEDULERS.register_module(module=lr_scheduler.CosineAnnealingLR)
SCHEDULERS.register_module(module=lr_scheduler.ChainedScheduler)
SCHEDULERS.register_module(module=lr_scheduler.SequentialLR)
SCHEDULERS.register_module(module=lr_scheduler.ReduceLROnPlateau)
SCHEDULERS.register_module(module=lr_scheduler.CyclicLR)
SCHEDULERS.register_module(module=lr_scheduler.OneCycleLR)
SCHEDULERS.register_module(module=lr_scheduler.CosineAnnealingWarmRestarts)
