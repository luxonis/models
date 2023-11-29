import os

import optuna
import pytorch_lightning as pl
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.utilities import rank_zero_only  # type: ignore

from luxonis_train.callbacks import LuxonisProgressBar
from luxonis_train.models import LuxonisModel
from luxonis_train.utils.tracker import LuxonisTrackerPL

from .core import Core


class Tuner(Core):
    def __init__(self, cfg: str | dict, args: list[str] | tuple[str, ...] | None):
        """Main API which is used to perform hyperparameter tunning.

        Args:
            cfg (Union[str, dict]): path to config file or config dict used to setup training
            args (Optional[dict]): argument dict provided through command line, used for config overriding
        """
        super().__init__(cfg, args)

    def tune(self):
        """Runs Optuna tunning of hyperparameters."""

        pruner = (
            optuna.pruners.MedianPruner()
            if self.cfg.get("tuner.use_pruner")
            else optuna.pruners.NopPruner()
        )

        storage = None
        if self.cfg.tuner.storage.active:
            if self.cfg.tuner.storage.storage_type == "local":
                storage = "sqlite:///study_local.db"
            elif self.cfg.tuner.storage.storage_type == "remote":
                storage = "postgresql://{}:{}@{}:{}/{}".format(
                    os.environ["POSTGRES_USER"],
                    os.environ["POSTGRES_PASSWORD"],
                    os.environ["POSTGRES_HOST"],
                    os.environ["POSTGRES_PORT"],
                    os.environ["POSTGRES_DB"],
                )
            else:
                raise KeyError(
                    f"Storage type '{self.cfg.tuner.storage.storage_type}' "
                    "not supported. Choose one of ['local', 'remote']"
                )

        study = optuna.create_study(
            study_name=self.cfg.tuner.study_name,
            storage=storage,
            direction="minimize",
            pruner=pruner,
            load_if_exists=True,
        )

        study.optimize(
            self._objective,
            n_trials=self.cfg.tuner.n_trials,
            timeout=self.cfg.tuner.timeout,
        )

    def _objective(self, trial: optuna.trial.Trial):
        """Objective function used to optimize Optuna study."""
        rank = rank_zero_only.rank
        cfg_logger = self.cfg.logger
        logger_params = cfg_logger.model_dump()
        logger_params.pop("logged_hyperparams")
        logger = LuxonisTrackerPL(
            rank=rank,
            mlflow_tracking_uri=os.getenv(
                "MLFLOW_TRACKING_URI"
            ),  # read seperately from env vars
            is_sweep=True,
            **logger_params,
        )
        run_save_dir = os.path.join(cfg_logger.save_directory, logger.run_name)

        # get curr trial params and update config
        curr_params = self._get_trial_params(trial)
        for key, value in curr_params.items():
            self.cfg.override_config({key: value})

        logger.log_hyperparams(curr_params)  # log curr trial params

        # save current config to logger directory
        self.cfg.save_data(os.path.join(run_save_dir, "config.yaml"))

        lightning_module = LuxonisModel(
            cfg=self.cfg,
            dataset_metadata=self.dataset_metadata,
            save_dir=run_save_dir,
            input_shape=self.loader_train.input_shape,
        )
        pruner_callback = PyTorchLightningPruningCallback(
            trial, monitor="val_loss/loss"
        )
        callbacks: list[pl.Callback] = (
            [LuxonisProgressBar()] if self.cfg.train.use_rich_text else []
        )
        callbacks.append(pruner_callback)
        pl_trainer = pl.Trainer(
            accelerator=self.cfg.trainer.accelerator,
            devices=self.cfg.trainer.devices,
            strategy=self.cfg.trainer.strategy,
            logger=logger,
            max_epochs=self.cfg.train.epochs,
            accumulate_grad_batches=self.cfg.train.accumulate_grad_batches,
            check_val_every_n_epoch=self.cfg.train.validation_interval,
            num_sanity_val_steps=self.cfg.trainer.num_sanity_val_steps,
            profiler=self.cfg.trainer.profiler,
            callbacks=callbacks,
        )

        pl_trainer.fit(
            lightning_module, self.pytorch_loader_train, self.pytorch_loader_val
        )
        pruner_callback.check_pruned()

        return pl_trainer.callback_metrics["val/loss"].item()

    def _get_trial_params(self, trial: optuna.trial.Trial):
        """Get trial params based on specified config."""
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

        if len(new_params) == 0:
            raise ValueError(
                "No paramteres to tune. Specify them under `tuner.params`."
            )
        return new_params
