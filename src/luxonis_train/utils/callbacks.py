from typing import Any, Dict, Optional, List
import warnings
import pytorch_lightning as pl
from pytorch_lightning.callbacks import RichProgressBar, BaseFinetuning
from rich.table import Table
from torch import Tensor
from torch.optim.optimizer import Optimizer

from luxonis_train.utils.filesystem import LuxonisFileSystem


class LuxonisProgressBar(RichProgressBar):
    def __init__(self):
        """Custom rich text progress bar based on RichProgressBar from Pytorch Lightning"""
        super().__init__(leave=True)

    def print_single_line(self, text: str) -> None:
        self._console.print(f"[magenta]{text}[/magenta]")

    def get_metrics(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> Dict[str, int | str | float | Dict[str, float]]:
        # NOTE: there might be a cleaner way of doing this
        items = super().get_metrics(trainer, pl_module)
        if trainer.training:
            items["Loss"] = pl_module.training_step_outputs[-1]["loss"].item()
        return items

    def print_results(
        self, stage: str, loss: float, metrics: Dict[str, Dict[str, Tensor]]
    ) -> None:
        """Prints results to the console using rich text"""

        self._console.rule(stage, style="bold magenta")
        self._console.print(f"[bold magenta]Loss:[/bold magenta] [white]{loss}[/white]")
        self._console.print(f"[bold magenta]Metrics:[/bold magenta]")
        for head in metrics:
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Metric name", style="magenta")
            table.add_column(head)
            for metric_name in metrics[head]:
                value = "{:.5f}".format(metrics[head][metric_name].cpu().item())
                table.add_row(metric_name, value)
            self._console.print(table)
        self._console.rule(style="bold magenta")


class ModuleFreezer(BaseFinetuning):
    def __init__(self, freeze_info: Dict[str, bool | List[bool]]):
        """Callback that freezes parts of the model based on provided dict

        Args:
            freeze_info (Dict[str, bool]): Dictionary where key is name of the
                model's part and value is bool flag for freezing
        """
        super().__init__()

        self.freeze_info = freeze_info

    def freeze_before_training(self, pl_module: pl.LightningModule) -> None:
        for key, value in self.freeze_info.items():
            if key == "backbone" and value:
                self.freeze(pl_module.model.backbone, train_bn=False)
            elif key == "neck" and value:
                if pl_module.model.neck:
                    self.freeze(pl_module.model.neck, train_bn=False)
                else:
                    warnings.warn(
                        "Skipping neck freezing as model doesn't have a neck."
                    )
            elif key == "heads":
                for i, v in enumerate(value):
                    if v:
                        self.freeze(pl_module.model.heads[i], train_bn=False)

    def finetune_function(
        self, pl_module: pl.LightningModule, epoch: int, optimizer: Optimizer
    ) -> None:
        pass


class AnnotationChecker(pl.Callback):
    def __init__(self):
        """Callback that checks if all annotatios that are required by the heads are present in the label dict"""
        super().__init__()
        self.first_train_batch = True
        self.first_val_batch = True
        self.first_test_batch = True

    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        super().on_train_batch_start(trainer, pl_module, batch, batch_idx)
        if self.first_train_batch:
            pl_module.model.check_annotations(batch[1])
            self.first_train_batch = False

    def on_validation_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        super().on_validation_batch_start(
            trainer, pl_module, batch, batch_idx, dataloader_idx
        )
        if self.first_val_batch:
            pl_module.model.check_annotations(batch[1])
            self.first_val_batch = False

    def on_test_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        super().on_test_batch_start(
            trainer, pl_module, batch, batch_idx, dataloader_idx
        )
        if self.first_test_batch:
            pl_module.model.check_annotations(batch[1])
            self.first_test_batch = False


class TestOnTrainEnd(pl.Callback):
    """Callback that performs test on pl_module when train ends"""

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        from luxonis_train.utils.config import Config
        from torch.utils.data import DataLoader
        from luxonis_ml.data import LuxonisDataset, BucketType, BucketStorage
        from luxonis_ml.loader import LuxonisLoader, ValAugmentations

        cfg = Config()
        with LuxonisDataset(
            team_id=self.cfg.get("dataset.team_id"),
            dataset_id=self.cfg.get("dataset.dataset_id"),
            bucket_type=eval(self.cfg.get("dataset.bucket_type")),
            bucket_storage=eval(self.cfg.get("dataset.bucket_storage")),
        ) as dataset:
            loader_test = LuxonisLoader(
                dataset,
                view=cfg.get("dataset.test_view"),
                augmentations=ValAugmentations(
                    image_size=self.cfg.get("train.preprocessing.train_image_size"),
                    augmentations=self.cfg.get("train.preprocessing.augmentations"),
                    train_rgb=self.cfg.get("train.preprocessing.train_rgb"),
                    keep_aspect_ratio=self.cfg.get(
                        "train.preprocessing.keep_aspect_ratio"
                    ),
                ),
            )
            pytorch_loader_test = DataLoader(
                loader_test,
                batch_size=cfg.get("train.batch_size"),
                num_workers=cfg.get("train.num_workers"),
                collate_fn=loader_test.collate_fn,
            )
            trainer.test(pl_module, pytorch_loader_test)


class ExportOnTrainEnd(pl.Callback):
    def __init__(self, override_upload_directory: bool):
        """Callback that performs export on train end with best weights according to the validation loss

        Args:
            override_upload_directory (bool): If True override upload_directory in Exporter with
                currently active MLFlow run (if present)
        """
        super().__init__()
        self.override_upload_directory = override_upload_directory

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        from luxonis_train.utils.config import Config
        from luxonis_train.core import Exporter

        cfg = Config()

        model_checkpoint_callbacks = [
            c for c in trainer.callbacks if isinstance(c, pl.callbacks.ModelCheckpoint)
        ]
        # NOTE: assume that first checkpoint callback is based on val loss
        best_model_path = model_checkpoint_callbacks[0].best_model_path

        # override export_weights path with path to currently best weights
        override = f"exporter.export_weights {best_model_path}"
        if self.override_upload_directory:
            if cfg.get("logger.is_mlflow"):
                new_upload_directory = (
                    f"mlflow://{trainer.logger.project_id}/{trainer.logger.run_id}"
                )
                override += f" exporter.upload.upload_directory {new_upload_directory}"
            else:
                warnings.warn(
                    "`override_upload_directory` specified but no MLFlow active run, skipping."
                )

        exporter = Exporter(
            cfg="", args={"override": override}  # singleton instance already present
        )
        exporter.export()


class UploadCheckpointOnTrainEnd(pl.Callback):
    def __init__(self, upload_directory: Optional[str] = None):
        """Callback that uploads best checkpoint to specified storage according to the validation loss

        Args:
            upload_directory (str): Path used as upload directory
        """
        super().__init__()
        if upload_directory is None:
            raise ValueError(
                "Should specify `upload_directory` if using `upload_checkpoint_on_finish` callback."
            )
        self.fs = LuxonisFileSystem(
            upload_directory, allow_active_mlflow_run=True, allow_local=False
        )

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        print(f"Started checkpoint upload to {self.fs.full_path()}...")
        model_checkpoint_callbacks = [
            c for c in trainer.callbacks if isinstance(c, pl.callbacks.ModelCheckpoint)
        ]
        # NOTE: assume that first checkpoint callback is based on val loss
        local_path = model_checkpoint_callbacks[0].best_model_path
        self.fs.put_file(
            local_path=local_path,
            remote_path=local_path.split("/")[-1],
            mlflow_instance=trainer.logger.experiment.get("mlflow", None),
        )
        print("Checkpoint upload finished")
