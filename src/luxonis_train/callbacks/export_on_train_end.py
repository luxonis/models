from pathlib import Path

import pytorch_lightning as pl

from luxonis_train.utils.registry import CALLBACKS


@CALLBACKS.register_module()
class ExportOnTrainEnd(pl.Callback):
    def __init__(self, override_upload_directory: bool):
        """Callback that performs export on train end with best weights according to the
        validation loss.

        Args:
            override_upload_directory (bool): If True override upload_directory
            in Exporter with
                currently active MLFlow run (if present)
        """
        super().__init__()
        self.override_upload_directory = override_upload_directory

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        from luxonis_train.core.exporter import Exporter

        model_checkpoint_callbacks = [
            c
            for c in trainer.callbacks  # type: ignore
            if isinstance(c, pl.callbacks.ModelCheckpoint)  # type: ignore
        ]
        # NOTE: assume that first checkpoint callback is based on val loss
        best_model_path = model_checkpoint_callbacks[0].best_model_path
        exporter = Exporter(
            cfg=pl_module.cfg,
            opts=["model.weights", best_model_path],
        )
        onnx_path = str(Path(best_model_path).parent.with_suffix(".onnx"))
        exporter.export(onnx_path=onnx_path)
