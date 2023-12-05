import pytorch_lightning as pl
from pytorch_lightning.callbacks import BaseFinetuning
from torch import nn
from torch.optim.optimizer import Optimizer


class ModuleFreezer(BaseFinetuning):
    """Callback that freezes parts of the model."""

    def __init__(self, frozen_modules: list[nn.Module]):
        """Constructs `ModuleFreezer`.

        Args:
            frozen_modules (list[nn.Module]): List of modules to freeze.
        """
        super().__init__()
        self.frozen_modules = frozen_modules

    def freeze_before_training(self, _: pl.LightningModule) -> None:
        for module in self.frozen_modules:
            self.freeze(module, train_bn=False)

    def finetune_function(
        self, pl_module: pl.LightningModule, epoch: int, optimizer: Optimizer
    ) -> None:
        # Called on every train epoch start. Used to unfreeze frozen modules.
        # TODO: Implement unfreezing and support in config.
        ...
