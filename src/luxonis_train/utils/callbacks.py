import pytorch_lightning as pl
from pytorch_lightning.callbacks import RichProgressBar
from rich.table import Table


class LuxonisProgressBar(RichProgressBar):
    """Custom rich text progress bar based on RichProgressBar from Pytorch Lightning"""

    def __init__(self):
        # TODO: play with values to create custom output
        # from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
        # progress_bar = RichProgressBar(
        #     theme = RichProgressBarTheme(
        #     description="green_yellow",
        #     progress_bar="green1",
        #     progress_bar_finished="green1",
        #     batch_progress="green_yellow",
        #     time="gray82",
        #     processing_speed="grey82",
        #     metrics="yellow1"
        #     )
        # )

        super().__init__(leave=True)

    def print_single_line(self, text: str):
        self._console.print(f"[magenta]{text}[/magenta]")

    def get_metrics(self, trainer, pl_module):
        # NOTE: there might be a cleaner way of doing this
        items = super().get_metrics(trainer, pl_module)
        if trainer.training:
            items["Loss"] = pl_module.training_step_outputs[-1]["loss"].item()
        return items

    def print_results(self, stage: str, loss: float, metrics: dict):
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


class TestOnTrainEnd(pl.Callback):
    """Callback that performs test on pl_module when train ends"""

    def on_train_end(self, trainer, pl_module):
        from torch.utils.data import DataLoader
        from luxonis_ml.data import LuxonisDataset, BucketType, BucketStorage
        from luxonis_ml.loader import LuxonisLoader, ValAugmentations
        from luxonis_train.utils.config import Config

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
    """Callback that performs export on train end with best weights according to the validation loss"""

    def on_train_end(self, trainer, pl_module):
        from luxonis_train.core import Exporter

        model_checkpoint_callbacks = [
            c for c in trainer.callbacks if isinstance(c, pl.callbacks.ModelCheckpoint)
        ]
        # NOTE: assume that first checkpoint callback is based on val loss
        best_model_path = model_checkpoint_callbacks[0].best_model_path

        # override export_weights path with path to currently best weights
        override = f"exporter.export_weights {best_model_path}"
        exporter = Exporter(
            cfg="", args={"override": override}  # singleton instance already present
        )
        exporter.export()
