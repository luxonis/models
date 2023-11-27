import os
import warnings
from typing import Any

import pytorch_lightning as pl
import rich.traceback
import torch
from dotenv import load_dotenv
from luxonis_ml.data import LuxonisDataset, TrainAugmentations, ValAugmentations
from pydantic import ValidationError
from pytorch_lightning.utilities import rank_zero_only  # type: ignore
from rich import print

from luxonis_train.callbacks import LuxonisProgressBar
from luxonis_train.utils.config import Config
from luxonis_train.utils.general import DatasetMetadata
from luxonis_train.utils.loaders import LuxonisLoaderTorch, collate_fn
from luxonis_train.utils.tracker import LuxonisTrackerPL


class LuxonisCore:
    """Common logic of the core components.

    This class contains common logic of the core components (trainer, evaluator,
    exporter, etc.).
    """

    def __init__(
        self,
        cfg: str | dict[str, Any] | Config,
        opts: list[str] | tuple[str, ...] | None = None,
    ):
        """Constructs a new Core instance.

        Loads the config and initializes datasets, dataloaders, augmentations,
        lightning components, etc.

        Args:
            cfg (str | dict): path to config file or config dict used to setup training
            opts (list[str]): argument dict provided through command line, used for config overriding
        """

        if isinstance(cfg, Config):
            self.cfg = cfg
        else:
            try:
                self.cfg = Config(cfg)  # type: ignore
            except ValidationError as e:
                print(e.errors())
                raise e

        opts = opts or []

        if opts:
            if len(opts) % 2 != 0:
                raise ValueError("Override options should be a list of key-value pairs")
            self.cfg.override_config(dict(zip(opts[::2], opts[1::2])))

        if self.cfg.train.use_rich_text:
            rich.traceback.install(suppress=[pl, torch])

        self.rank = rank_zero_only.rank

        cfg_logger = self.cfg.logger

        load_dotenv()  # loads env variables for mlflow logging
        logger_params = cfg_logger.model_dump()
        logger_params.pop("logged_hyperparams")
        self.logger = LuxonisTrackerPL(
            rank=self.rank,
            mlflow_tracking_uri=os.getenv(
                "MLFLOW_TRACKING_URI"
            ),  # read separately from env vars
            **logger_params,
        )

        self.run_save_dir = os.path.join(
            cfg_logger.save_directory, self.logger.run_name
        )

        self.train_augmentations = TrainAugmentations(
            image_size=self.cfg.train.preprocessing.train_image_size,
            augmentations=[
                i.model_dump() for i in self.cfg.train.preprocessing.augmentations
            ],
            train_rgb=self.cfg.train.preprocessing.train_rgb,
            keep_aspect_ratio=self.cfg.train.preprocessing.keep_aspect_ratio,
        )
        self.val_augmentations = ValAugmentations(
            image_size=self.cfg.train.preprocessing.train_image_size,
            augmentations=[
                i.model_dump() for i in self.cfg.train.preprocessing.augmentations
            ],
            train_rgb=self.cfg.train.preprocessing.train_rgb,
            keep_aspect_ratio=self.cfg.train.preprocessing.keep_aspect_ratio,
        )

        self.pl_trainer = pl.Trainer(
            accelerator=self.cfg.trainer.accelerator,
            devices=self.cfg.trainer.devices,
            strategy=self.cfg.trainer.strategy,
            logger=self.logger,
            max_epochs=self.cfg.train.epochs,
            accumulate_grad_batches=self.cfg.train.accumulate_grad_batches,
            check_val_every_n_epoch=self.cfg.train.validation_interval,
            num_sanity_val_steps=self.cfg.trainer.num_sanity_val_steps,
            profiler=self.cfg.trainer.profiler,  # for debugging purposes,
            # NOTE: this is likely PL bug,
            # should be configurable inside configure_callbacks(),
            callbacks=LuxonisProgressBar() if self.cfg.train.use_rich_text else None,
        )
        self.dataset = LuxonisDataset(
            dataset_name=self.cfg.dataset.dataset_name,
            team_id=self.cfg.dataset.team_id,
            dataset_id=self.cfg.dataset.dataset_id,
            bucket_type=self.cfg.dataset.bucket_type,
            bucket_storage=self.cfg.dataset.bucket_storage,
        )

        self.dataset_metadata = DatasetMetadata.from_dataset(self.dataset)

        self.loader_train = LuxonisLoaderTorch(
            self.dataset,
            view=self.cfg.dataset.train_view,
            augmentations=self.train_augmentations,
        )
        self.loader_val = LuxonisLoaderTorch(
            self.dataset,
            view=self.cfg.dataset.val_view,
            augmentations=self.val_augmentations,
        )
        self.loader_test = LuxonisLoaderTorch(
            self.dataset,
            view=self.cfg.dataset.test_view,
            augmentations=self.val_augmentations,
        )

        self.pytorch_loader_val = torch.utils.data.DataLoader(
            self.loader_val,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
            collate_fn=collate_fn,
        )
        self.pytorch_loader_test = torch.utils.data.DataLoader(
            self.loader_test,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
            collate_fn=collate_fn,
        )
        sampler = None
        if self.cfg.train.use_weighted_sampler:
            classes_count = self.dataset.get_classes()[1]
            if len(classes_count) == 0:
                warnings.warn(
                    "WeightedRandomSampler only available for classification tasks. Using default sampler instead."
                )
            else:
                weights = [1 / i for i in classes_count.values()]
                num_samples = sum(classes_count.values())
                sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples)

        self.pytorch_loader_train = torch.utils.data.DataLoader(
            self.loader_train,
            shuffle=True,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
            collate_fn=collate_fn,
            drop_last=self.cfg.train.skip_last_batch,
            sampler=sampler,
        )
        self.error_message = None

        self.cfg.save_data(os.path.join(self.run_save_dir, "config.yaml"))

    def set_train_augmentations(self, aug: TrainAugmentations) -> None:
        """Sets augmentations used for training dataset."""
        self.train_augmentations = aug

    def set_val_augmentations(self, aug: ValAugmentations) -> None:
        """Sets augmentations used for validation dataset."""
        self.val_augmentations = aug

    def set_test_augmentations(self, aug: ValAugmentations) -> None:
        """Sets augmentations used for test dataset."""
        self.test_augmentations = aug

    @rank_zero_only
    def get_save_dir(self):
        """Return path to directory where checkpoints are saved.

        Returns:
            str: Save directory path
        """
        return self.run_save_dir

    @rank_zero_only
    def get_error_message(self):
        """Return error message if one occurs while running in thread, otherwise None.

        Returns:
            str or None: Error message
        """
        return self.error_message

    @rank_zero_only
    def get_min_loss_checkpoint_path(self):
        """Return best checkpoint path with respect to minimal validation loss.

        Returns:
            str: Path to best checkpoint with respect to minimal validation loss
        """
        return self.pl_trainer.checkpoint_callbacks[0].best_model_path  # type: ignore

    @rank_zero_only
    def get_best_metric_checkpoint_path(self):
        """Return best checkpoint path with respect to best validation metric.

        Returns:
            str: Path to best checkpoint with respect to best validation loss
        """
        return self.pl_trainer.checkpoint_callbacks[1].best_model_path  # type: ignore
