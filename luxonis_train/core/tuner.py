import pytorch_lightning as pl
import optuna
import warnings
import torch
import os
from typing import Union
from dotenv import load_dotenv
from copy import deepcopy
from pytorch_lightning.utilities import rank_zero_only
from optuna.integration import PyTorchLightningPruningCallback

from luxonis_train.utils.config import Config
from luxonis_train.utils.callbacks import LuxonisProgressBar
from luxonis_train.utils.augmentations import TrainAugmentations, ValAugmentations
from luxonis_train.models import ModelLightningModule
from luxonis_ml import *

class Tuner:
    def __init__(self, cfg: Union[str, dict], args: dict = None):
        self.cfg_data = cfg
        self.args = args

    def tune(self):
        pruner = optuna.pruners.MedianPruner()
        storage = "sqlite:///example.db"
        study = optuna.create_study(
            study_name="test_study_1",
            storage=storage,
            direction="minimize",
            pruner=pruner,
            load_if_exists=True,
        )
        study.optimize(self._objective, n_trials=3, timeout=600)

    def _objective(self, trial: optuna.trial.Trial):
        # Config.clear_instance() # TODO: check if this is needed because config is singleton
        self.cfg = Config(self.cfg_data)
        if self.args and self.args["override"]:
            self.cfg.override_config(self.args["override"])
        
        load_dotenv() # loads env variables for mlflow logging
        rank = rank_zero_only.rank
        cfg_logger = self.cfg.get("logger")
        logger_params = deepcopy(cfg_logger.copy())
        logger_params.pop("logged_hyperparams")
        logger = LuxonisTrackerPL(rank=rank, **logger_params)
        run_save_dir = os.path.join(cfg_logger["save_directory"], logger.run_name)

        # trial specific parameters
        batch_size = trial.suggest_int("batch_size", 4, 8)
        self.cfg.override_config(
            f"train.batch_size {batch_size}"
        )

        # log trial specific parameters
        logger.log_hyperparams({"batch_size":batch_size})
        # save current config to logger directory
        self.cfg.save_data(os.path.join(run_save_dir, "config.yaml"))

        pruner_callback = PyTorchLightningPruningCallback(trial, monitor="val_loss/loss")

        lightning_module = ModelLightningModule(run_save_dir)
        pl_trainer = pl.Trainer(
            accelerator=self.cfg.get("trainer.accelerator"),
            devices=self.cfg.get("trainer.devices"),
            strategy=self.cfg.get("trainer.strategy"),
            logger=logger,
            max_epochs=self.cfg.get("train.epochs"),
            accumulate_grad_batches=self.cfg.get("train.accumulate_grad_batches"),
            check_val_every_n_epoch=self.cfg.get("train.validation_interval"),
            num_sanity_val_steps=self.cfg.get("trainer.num_sanity_val_steps"),
            profiler=self.cfg.get("trainer.profiler"), # for debugging purposes,
            callbacks=[
                LuxonisProgressBar() if self.cfg.get("train.use_rich_text") else None, # NOTE: this is likely PL bug, should be configurable inside configure_callbacks(),
                pruner_callback,
            ]
        )

        with LuxonisDataset(
            team_name=self.cfg.get("dataset.team_name"),
            dataset_name=self.cfg.get("dataset.dataset_name")
        ) as dataset:

            loader_train = LuxonisLoader(
                dataset,
                view=self.cfg.get("dataset.train_view"),
                augmentations=TrainAugmentations()
            )

            sampler = None
            if self.cfg.get("train.use_weighted_sampler"):
                classes_count = dataset.get_classes_count()
                if len(classes_count) == 0:
                    warnings.warn("WeightedRandomSampler only available for classification tasks. Using default sampler instead.")
                else:
                    weights = [1/i for i in classes_count.values()]
                    num_samples = sum(classes_count.values())
                    sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples)

            pytorch_loader_train = torch.utils.data.DataLoader(
                loader_train,
                batch_size=self.cfg.get("train.batch_size"),
                num_workers=self.cfg.get("train.num_workers"),
                collate_fn=loader_train.collate_fn,
                drop_last=self.cfg.get("train.skip_last_batch"),
                sampler=sampler
            )

            loader_val = LuxonisLoader(
                dataset,
                view=self.cfg.get("dataset.val_view"),
                augmentations=ValAugmentations()
            )
            pytorch_loader_val = torch.utils.data.DataLoader(
                loader_val,
                batch_size=self.cfg.get("train.batch_size"),
                num_workers=self.cfg.get("train.num_workers"),
                collate_fn=loader_val.collate_fn
            )

            pl_trainer.fit(lightning_module, pytorch_loader_train, pytorch_loader_val)
            pruner_callback.check_pruned()

            return pl_trainer.callback_metrics["val_loss/loss"].item()