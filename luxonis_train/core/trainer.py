import threading
from typing import Any, Literal

from lightning.pytorch.utilities import rank_zero_only  # type: ignore

from luxonis_train.models import LuxonisModel
from luxonis_train.utils.config import Config

from .core import Core


class Trainer(Core):
    """Main API which is used to create the model, setup pytorch lightning environment
    and perform training based on provided arguments and config."""

    def __init__(
        self,
        cfg: str | dict[str, Any] | Config,
        opts: list[str] | tuple[str, ...] | dict[str, Any] | None = None,
    ):
        """Constructs a new Trainer instance.

        Args:
            cfg (str | dict): path to config file or config dict used to setup training
            args (dict | None): argument dict provided through command line, used for config overriding
        """
        super().__init__(cfg, opts)

        self.lightning_module = LuxonisModel(
            cfg=self.cfg,
            dataset_metadata=self.dataset_metadata,
            save_dir=self.run_save_dir,
            input_shape=self.loader_train.input_shape,
        )

    def train(self, new_thread: bool = False) -> None:
        """Runs training.

        Args:
            new_thread (bool, optional): Runs training in new thread if set to True. Defaults to False.
        """
        if not new_thread:
            self.logger.info(f"Checkpoints will be saved in: {self.get_save_dir()}")
            self.logger.info("Starting training...")
            self.pl_trainer.fit(
                self.lightning_module,
                self.pytorch_loader_train,
                self.pytorch_loader_val,
            )
            self.logger.info("Training finished")
            self.logger.info(f"Checkpoints saved in: {self.get_save_dir()}")
        else:
            # Every time exception happens in the Thread, this hook will activate
            def thread_exception_hook(args):
                self.error_message = str(args.exc_value)

            threading.excepthook = thread_exception_hook

            self.thread = threading.Thread(
                target=self.pl_trainer.fit,
                args=(
                    self.lightning_module,
                    self.pytorch_loader_train,
                    self.pytorch_loader_val,
                ),
                daemon=True,
            )
            self.thread.start()

    def test(
        self, new_thread: bool = False, view: Literal["train", "val", "test"] = "test"
    ) -> None:
        """Runs testing
        Args:
            new_thread (bool, optional): Runs training in new thread if set to True.
                Defaults to False.
        """

        if view == "test":
            loader = self.pytorch_loader_test
        elif view == "val":
            loader = self.pytorch_loader_val
        elif view == "train":
            loader = self.pytorch_loader_train

        if not new_thread:
            self.pl_trainer.test(self.lightning_module, loader)
        else:
            self.thread = threading.Thread(
                target=self.pl_trainer.test,
                args=(self.lightning_module, loader),
                daemon=True,
            )
            self.thread.start()

    @rank_zero_only
    def get_status(self) -> tuple[int, int]:
        """Get current status of training.

        Returns:
            tuple[int, int]: First element is current epoch, second element is total
              number of epochs.
        """
        return self.lightning_module.get_status()

    @rank_zero_only
    def get_status_percentage(self) -> float:
        """Return percentage of current training, takes into account early stopping.

        Returns:
            float: Percentage of current training in range 0-100.
        """
        return self.lightning_module.get_status_percentage()
