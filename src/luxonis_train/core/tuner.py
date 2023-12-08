import os.path as osp
from typing import Any

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

    def tune(self) -> None:
        """Runs Optuna tunning of hyperparameters."""

        pruner = (
            optuna.pruners.MedianPruner()
            if self.cfg.tuner.use_pruner
            else optuna.pruners.NopPruner()
        )

        storage = None
        if self.cfg.tuner.storage.active:
            if self.cfg.tuner.storage.storage_type == "local":
                storage = "sqlite:///study_local.db"
            else:
                storage = "postgresql://{}:{}@{}:{}/{}".format(
                    self.cfg.ENVIRON.POSTGRES_USER,
                    self.cfg.ENVIRON.POSTGRES_PASSWORD,
                    self.cfg.ENVIRON.POSTGRES_HOST,
                    self.cfg.ENVIRON.POSTGRES_PORT,
                    self.cfg.ENVIRON.POSTGRES_DB,
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

    def _objective(self, trial: optuna.trial.Trial) -> float:
        """Objective function used to optimize Optuna study."""
        rank = rank_zero_only.rank
        cfg_tracker = self.cfg.tracker
        tracker_params = cfg_tracker.model_dump()
        tracker = LuxonisTrackerPL(
            rank=rank,
            mlflow_tracking_uri=self.cfg.ENVIRON.MLFLOW_TRACKING_URI,
            is_sweep=True,
            **tracker_params,
        )
        run_save_dir = osp.join(cfg_tracker.save_directory, tracker.run_name)

        curr_params = self._get_trial_params(trial)
        self.cfg.override_config(curr_params)

        tracker.log_hyperparams(curr_params)

        self.cfg.save_data(osp.join(run_save_dir, "config.yaml"))

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
            [LuxonisProgressBar()] if self.cfg.use_rich_text else []
        )
        callbacks.append(pruner_callback)
        pl_trainer = pl.Trainer(
            accelerator=self.cfg.trainer.accelerator,
            devices=self.cfg.trainer.devices,
            strategy=self.cfg.trainer.strategy,
            logger=tracker,
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

        if "val/loss" not in pl_trainer.callback_metrics:
            raise ValueError(
                "No validation loss found. "
                "This can happen if `TestOnTrainEnd` callback is used."
            )

        return pl_trainer.callback_metrics["val/loss"].item()

    def _get_trial_params(self, trial: optuna.trial.Trial) -> dict[str, Any]:
        """Get trial params based on specified config."""
        cfg_tuner = self.cfg.tuner.params
        new_params = {}
        for key, value in cfg_tuner.items():
            key_info = key.split("_")
            key_name = "_".join(key_info[:-1])
            key_type = key_info[-1]
            print(key_name)
            match key_type, value:
                case "categorical", list(lst):
                    new_value = trial.suggest_categorical(key_name, lst)
                case "float", [float(low), float(high), *step]:
                    step = step[0] if step else None
                    if step is not None and not isinstance(step, float):
                        raise ValueError(
                            f"Step for float type must be float, but got {step}"
                        )
                    new_value = trial.suggest_float(key_name, low, high, step=step)
                case "int", [int(low), int(high), *step]:
                    step = step[0] if step else 1
                    if not isinstance(step, int):
                        raise ValueError(
                            f"Step for int type must be int, but got {step}"
                        )
                    new_value = trial.suggest_int(key_name, low, high, step=step)
                case "loguniform", [float(low), float(high)]:
                    new_value = trial.suggest_loguniform(key_name, low, high)
                case "uniform", [float(low), float(high)]:
                    new_value = trial.suggest_uniform(key_name, low, high)
                case _, _:
                    raise KeyError(
                        f"Combination of {key_type} and {value} not supported"
                    )

            new_params[key_name] = new_value

        if len(new_params) == 0:
            raise ValueError(
                "No paramteres to tune. Specify them under `tuner.params`."
            )
        return new_params
