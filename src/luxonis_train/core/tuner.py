import pytorch_lightning as pl
import warnings
import torch
import os
import optuna
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
        """Main API which is used to perform hyperparameter tunning

        Args:
            cfg (Union[str, dict]): path to config file or config dict used to setup training
            args (dict, optional): argument dict provided through command line, used for config overriding
        """
        self.cfg_data = cfg
        self.args = args
        load_dotenv()

    def tune(self):
        """ Runs Optuna tunning of hyperparameters """
        self.cfg = Config(self.cfg_data)
        if self.args and self.args["override"]:
            self.cfg.override_config(self.args["override"])
        self.cfg.validate_config_tuner()

        pruner = optuna.pruners.MedianPruner() if self.cfg.get("tuner.use_pruner") \
            else optuna.pruners.NopPruner()
        
        storage = None
        if self.cfg.get("tuner.storage.active"):
            if self.cfg.get("tuner.storage.type") == "local":
                storage = "sqlite:///study_local.db"
            elif self.cfg.get("tuner.storage.type") == "remote":
                storage = "postgresql://{}:{}@{}:{}/{}".format(
                    os.environ["POSTGRES_USER"],
                    os.environ["POSTGRES_PASSWORD"],
                    os.environ["POSTGRES_HOST"],
                    os.environ["POSTGRES_PORT"],
                    os.environ["POSTGRES_DB"],
                )
            else:
                raise KeyError(f"Storage type '{self.cfg.get('tuner.storage.type')}'"+
                    "not supported. Choose one of ['local', 'remote']")

        study = optuna.create_study(
            study_name=self.cfg.get("tuner.study_name"),
            storage=storage,
            direction="minimize",
            pruner=pruner,
            load_if_exists=True,
        )

        study.optimize(
            self._objective, 
            n_trials=self.cfg.get("tuner.n_trials"),
            timeout=self.cfg.get("tuner.timeout")
        )

    def _objective(self, trial: optuna.trial.Trial):
        """ Objective function used to optimize Optuna study """
        # TODO: check if this is even needed needed because config is singleton
        # Config.clear_instance() 
        self.cfg = Config(self.cfg_data)
        if self.args and self.args["override"]:
            self.cfg.override_config(self.args["override"])
        self.cfg.validate_config_tuner()
        
        rank = rank_zero_only.rank
        cfg_logger = self.cfg.get("logger")
        logger_params = deepcopy(cfg_logger.copy())
        logger_params.pop("logged_hyperparams")
        logger = LuxonisTrackerPL(rank=rank, is_sweep=True, **logger_params)
        run_save_dir = os.path.join(cfg_logger["save_directory"], logger.run_name)

        # get curr trial params and update config
        curr_params = self._get_trial_params(trial)
        for key, value in curr_params.items():
            self.cfg.override_config(f"{key} {value}")

        logger.log_hyperparams(curr_params) # log curr trial params
        
        # save current config to logger directory
        self.cfg.save_data(os.path.join(run_save_dir, "config.yaml"))

        lightning_module = ModelLightningModule(run_save_dir)
        pruner_callback = PyTorchLightningPruningCallback(trial, monitor="val_loss/loss")
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
            team_name=self.cfg.get("dataset.team_id"),
            dataset_name=self.cfg.get("dataset.dataset_id"),
            bucket_type=self.cfg.get("dataset.bucket_type"),
            override_bucket_type=self.cfg.get("dataset.override_bucket_type")
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

    def _get_trial_params(self, trial: optuna.trial.Trial):
        """ Get trial params based on specified config """
        cfg_tuner = self.cfg.get("tuner.params")
        new_params = {}
        for key, value in cfg_tuner.items():
            key_info = key.split("_")
            key_name = "_".join(key_info[:-1])
            key_type = key_info[-1]
            
            if key_type == "categorical":
                # NOTE: might need to do some preprocessing if list doesn't only have strings
                new_value = trial.suggest_categorical(key_name, value)
            elif key_type == "float":
                new_value = trial.suggest_float(key_name, *value)
            elif key_type == "int":
                new_value = trial.suggest_int(key_name, *value)
            elif key_type == "loguniform":
                new_value = trial.suggest_loguniform(key_name, *value)
            elif key_type == "uniform":
                new_value = trial.suggest_uniform(key_name, *value)
            else:
                raise KeyError(f"Tunning type '{key_type}' not supported.")

            new_params[key_name] = new_value
        return new_params