import logging
from pathlib import Path
from typing import cast

import pytorch_lightning as pl

from luxonis_train.utils.registry import CALLBACKS
from luxonis_train.utils.tracker import LuxonisTrackerPL


@CALLBACKS.register_module()
class ExportOnTrainEnd(pl.Callback):
    def __init__(self, upload_to_mlflow: bool = False):
        """Callback that performs export on train end with best weights according to the
        validation loss.

        Args:
            upload_to_mlflow (bool, optional): If se to True, overrides the
              upload directory in Exporter with currently
              active MLFlow run (if present).
        """
        super().__init__()
        self.upload_to_mlflow = upload_to_mlflow

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        from luxonis_train.core.exporter import Exporter

        model_checkpoint_callbacks = [
            c
            for c in trainer.callbacks  # type: ignore
            if isinstance(c, pl.callbacks.ModelCheckpoint)  # type: ignore
        ]
        # NOTE: assume that first checkpoint callback is based on val loss
        best_model_path = model_checkpoint_callbacks[0].best_model_path
        opts = ["model.weights", best_model_path]
        if self.upload_to_mlflow:
            if pl_module.cfg.tracker.is_mlflow:
                tracker = cast(LuxonisTrackerPL, trainer.logger)
                new_upload_directory = f"mlflow://{tracker.project_id}/{tracker.run_id}"
                opts += ["exporter.upload_directory", new_upload_directory]
            else:
                logging.getLogger(__name__).warning(
                    "`upload_to_mlflow` is set to True, "
                    "but there is  no MLFlow active run, skipping."
                )
        exporter = Exporter(
            cfg=pl_module.cfg,
            opts=opts,
        )
        onnx_path = str(Path(best_model_path).parent.with_suffix(".onnx"))
        exporter.export(onnx_path=onnx_path)
