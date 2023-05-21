import os
import torch
import pytorch_lightning as pl
import threading
from copy import deepcopy
from typing import Union
from dotenv import load_dotenv
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.callbacks import RichProgressBar
from luxonis_ml import *

from luxonis_train.utils.callbacks import LuxonisProgressBar
from luxonis_train.models import ModelLightningModule
from luxonis_train.utils.config import Config
from luxonis_train.utils.augmentations import TrainAugmentations, ValAugmentations
from luxonis_train.utils.head_type import *

class Trainer:
    def __init__(self, cfg: Union[str, dict], args: dict = None):
        """Main API which is used to create the model, setup pytorch lightning environment
        and perform training based on provided arguments and config.

        Args:
            cfg (Union[str, dict]): path to config file or config dict used to setup training
            args (dict, optional): argument dict provided through command line, used for config overriding
        """

        self.cfg = Config(cfg)

        if args and args["override"]:
            self.cfg.override_config(args["override"])

        self.rank = rank_zero_only.rank

        cfg_logger = self.cfg.get("logger")
        hparams = {key: self.cfg.get(key) for key in cfg_logger["logged_hyperparams"]}

        load_dotenv() # loads env variables for mlflow logging
        logger_params = deepcopy(cfg_logger.copy())
        logger_params.pop("logged_hyperparams")
        logger = LuxonisTrackerPL(rank=self.rank, **logger_params)
        logger.log_hyperparams(hparams)

        self.run_save_dir = os.path.join(cfg_logger["save_directory"], logger.run_name)

        self.train_augmentations = None
        self.val_augmentations = None
        self.test_augmentations = None

        self.lightning_module = ModelLightningModule(self.run_save_dir)
        self.pl_trainer = pl.Trainer(
            accelerator=self.cfg.get("trainer.accelerator"),
            devices=self.cfg.get("trainer.devices"),
            strategy=self.cfg.get("trainer.strategy"),
            logger=logger,
            max_epochs=self.cfg.get("train.epochs"),
            accumulate_grad_batches=self.cfg.get("train.accumulate_grad_batches"),
            check_val_every_n_epoch=self.cfg.get("train.validation_interval"),
            num_sanity_val_steps=self.cfg.get("trainer.num_sanity_val_steps"),
            profiler=self.cfg.get("trainer.profiler"), # for debugging purposes,
            callbacks=LuxonisProgressBar() if self.cfg.get("train.use_rich_text") else None # NOTE: this is likely PL bug, should be configurable inside configure_callbacks(),
        )
        self.error_message = None

    def train(self, new_thread: bool = False):
        """ Runs training

        Args:
            new_thread (bool, optional): Runs training in new thread if set to True. Defaults to False.
        """

        with LuxonisDataset(
            team_name=self.cfg.get("dataset.team_name"),
            dataset_name=self.cfg.get("dataset.dataset_name")
        ) as dataset:

            if self.train_augmentations == None:
                self.train_augmentations = TrainAugmentations()

            loader_train = LuxonisLoader(
                dataset,
                view=self.cfg.get("dataset.train_view"),
                augmentations=self.train_augmentations
            )
            pytorch_loader_train = torch.utils.data.DataLoader(
                loader_train,
                batch_size=self.cfg.get("train.batch_size"),
                num_workers=self.cfg.get("train.num_workers"),
                collate_fn=loader_train.collate_fn,
                drop_last=self.cfg.get("train.skip_last_batch")
            )

            if self.val_augmentations == None:
                self.val_augmentations = ValAugmentations()

            loader_val = LuxonisLoader(
                dataset,
                view=self.cfg.get("dataset.val_view"),
                augmentations=self.val_augmentations
            )
            pytorch_loader_val = torch.utils.data.DataLoader(
                loader_val,
                batch_size=self.cfg.get("train.batch_size"),
                num_workers=self.cfg.get("train.num_workers"),
                collate_fn=loader_val.collate_fn
            )

            if not new_thread:
                self.pl_trainer.fit(self.lightning_module, pytorch_loader_train, pytorch_loader_val)
                print(f"Checkpoints saved in: {self.get_save_dir()}")
            else:
                # Every time expection happens in the Thread, this hook will activate
                def thread_exception_hook(args):
                    self.error_message = str(args.exc_value)
                threading.excepthook = thread_exception_hook

                self.thread = threading.Thread(
                    target=self.pl_trainer.fit,
                    args=(self.lightning_module, pytorch_loader_train, pytorch_loader_val),
                    daemon=True
                )
                self.thread.start()

    def test(self, new_thread: bool = False):
        """ Runs testing
        Args:
            new_thread (bool, optional): Runs training in new thread if set to True. Defaults to False.
        """

        with LuxonisDataset(
            team_name=self.cfg.get("dataset.team_name"),
            dataset_name=self.cfg.get("dataset.dataset_name")
        ) as dataset:

            if self.test_augmentations == None:
                self.test_augmentations = ValAugmentations()

            loader_test = LuxonisLoader(
                dataset,
                view=self.cfg.get("dataset.test_view"),
                augmentations=self.test_augmentations
            )
            pytorch_loader_test = torch.utils.data.DataLoader(
                loader_test,
                batch_size=self.cfg.get("train.batch_size"),
                num_workers=self.cfg.get("train.num_workers"),
                collate_fn=loader_test.collate_fn
            )

            if not new_thread:
                self.pl_trainer.test(self.lightning_module, pytorch_loader_test)
            else:
                self.thread = threading.Thread(
                    target=self.pl_trainer.test,
                    args=(self.lightning_module, pytorch_loader_test),
                    daemon=True
                )
                self.thread.start()

    def override_loss(self, custom_loss: object, head_id: int):
        """ Overrides loss function for specific head_id with custom loss """
        if not 0 <= head_id < len(self.lightning_module.model.heads):
            raise ValueError("Provided 'head_id' outside of range.")
        self.lightning_module.losses[head_id] = custom_loss

    def override_train_augmentations(self, aug: object):
        """ Overrides augmentations used for trainig dataset """
        self.train_augmentations = aug

    def override_val_augmentations(self, aug: object):
        """ Overrides augmentations used for validation dataset """
        self.val_augmentations = aug

    def override_test_augmentations(self, aug: object):
        """ Overrides augmentations used for test dataset """
        self.test_augmentations = aug

    @rank_zero_only
    def get_status(self):
        """Get current status of training

        Returns:
            Tuple(int, int): First element is current epoch, second element is total number of epochs
        """
        return self.lightning_module.get_status()

    @rank_zero_only
    def get_status_percentage(self):
        """ Return percentage of current training, takes into account early stopping

        Returns:
            float: Percentage of current training in range 0-100
        """
        return self.lightning_module.get_status_percentage()

    @rank_zero_only
    def get_save_dir(self):
        """ Return path to directory where checkpoints are saved

        Returns:
            str: Save directory path
        """
        return self.run_save_dir

    @rank_zero_only
    def get_error_message(self):
        """ Return error message if one occures while running in thread, otherwise None

        Returns:
            str or None: Error message
        """
        return self.error_message

    @rank_zero_only
    def get_min_loss_checkpoint_path(self):
        """ Return best checkpoint path with respect to minimal validation loss

        Returns:
            str: Path to best checkpoint with respect to minimal validation loss
        """
        return self.pl_trainer.checkpoint_callbacks[0].best_model_path

    @rank_zero_only
    def get_best_metric_checkpoint_path(self):
        """ Return best checkpoint path with respect to best validation metric

        Returns:
            str: Path to best checkpoint with respect to best validation loss
        """
        return self.pl_trainer.checkpoint_callbacks[1].best_model_path
