import pytorch_lightning as pl
import torch
import cv2
import torch.nn as nn
import numpy as np
from pprint import pprint
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.utilities import rank_zero_only

from luxonis_train.models import Model
from luxonis_train.utils.config import ConfigHandler
from luxonis_train.utils.registry import LOSSES, CALLBACKS, OPTIMIZERS, SCHEDULERS
from luxonis_train.utils.metrics import init_metrics
from luxonis_train.utils.visualization import draw_outputs, draw_labels
from luxonis_train.utils.filesystem import LuxonisFileSystem
from luxonis_train.utils.callbacks import (
    AnnotationChecker,
    ModuleFreezer,
    MetadataLogger,
)


class ModelLightningModule(pl.LightningModule):
    def __init__(self, save_dir: str):
        """Main class used to build and train the model using Pytorch Lightning"""
        super().__init__()

        self.cfg = ConfigHandler()
        self.save_dir = save_dir
        self.model_name = self.cfg.get("model.name")

        self.model = Model()
        self.model.build_model()

        # for each head get its loss
        self.losses = nn.ModuleList()
        for i, head in enumerate(self.cfg.get("model.heads")):
            # pass config params + head instance attributes
            params = {
                **head.loss.params,
                "head_attributes": self.model.heads[i].__dict__,
            }
            self.losses.append(LOSSES.get(head.loss.name)(**params))

        # for each head initialize its metrics
        self.metrics = nn.ModuleDict()
        for i, curr_head in enumerate(self.model.heads):
            self.metrics[curr_head.get_name(i)] = init_metrics(curr_head)

        # load pretrained weights if defined
        if self.cfg.get("model.pretrained"):
            self.load_checkpoint(self.cfg.get("model.pretrained"))

        # lists for storing intermediate step outputs
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def configure_callbacks(self):
        """Configures Pytorch Lightning callbacks"""
        self.min_val_loss_checkpoints_path = f"{self.save_dir}/min_val_loss"
        self.best_val_metric_checkpoints_path = f"{self.save_dir}/best_val_metric"
        self.main_metric = self.model.heads[0].main_metric

        loss_checkpoint = ModelCheckpoint(
            monitor="val_loss/loss",
            dirpath=self.min_val_loss_checkpoints_path,
            filename="loss={val_loss/loss:.4f}_"
            + self.main_metric
            + "={val_metric/"
            + self.main_metric
            + ":.4f}_{epoch:02d}_"
            + self.model_name,
            auto_insert_metric_name=False,
            save_top_k=self.cfg.get("train.callbacks.checkpoint.save_top_k"),
            mode="min",
        )

        metric_checkpoint = ModelCheckpoint(
            monitor=f"val_metric/{self.main_metric}",
            dirpath=self.best_val_metric_checkpoints_path,
            filename=self.main_metric
            + "={val_metric/"
            + self.main_metric
            + ":.4f}_loss={val_loss/loss:.4f}_{epoch:02d}_"
            + self.model_name,
            auto_insert_metric_name=False,
            save_top_k=self.cfg.get("train.callbacks.checkpoint.save_top_k"),
            mode="max",
        )

        lr_monitor = LearningRateMonitor(logging_interval="step")
        annotation_checker = AnnotationChecker()
        module_freezer = ModuleFreezer(
            freeze_info=self.cfg.get("train.freeze_modules").model_dump()
        )
        metadata_logger = MetadataLogger()

        callbacks = [
            loss_checkpoint,
            metric_checkpoint,
            lr_monitor,
            annotation_checker,
            module_freezer,
            metadata_logger,
        ]

        # used if we want to perform fine-grained debugging
        if self.cfg.get("train.callbacks.use_device_stats_monitor"):
            from pytorch_lightning.callbacks import DeviceStatsMonitor

            device_stats = DeviceStatsMonitor()
            callbacks.append(device_stats)

        if self.cfg.get("train.callbacks.early_stopping.active"):
            from pytorch_lightning.callbacks import EarlyStopping

            cfg_early_stopping = self.cfg.get(
                "train.callbacks.early_stopping"
            ).model_dump()

            cfg_early_stopping.pop("active")
            early_stopping = EarlyStopping(**cfg_early_stopping)
            callbacks.append(early_stopping)

        if self.cfg.get("train.use_rich_text"):
            from pytorch_lightning.callbacks import RichModelSummary

            callbacks.append(RichModelSummary())

        if self.cfg.get("train.callbacks.test_on_finish"):
            from luxonis_train.utils.callbacks import TestOnTrainEnd

            callbacks.append(TestOnTrainEnd())

        if self.cfg.get("train.callbacks.export_on_finish.active"):
            from luxonis_train.utils.callbacks import ExportOnTrainEnd

            callbacks.append(
                ExportOnTrainEnd(
                    override_upload_directory=self.cfg.get(
                        "train.callbacks.export_on_finish.override_upload_directory"
                    )
                )
            )

        if self.cfg.get("train.callbacks.upload_checkpoint_on_finish.active"):
            from luxonis_train.utils.callbacks import UploadCheckpointOnTrainEnd

            callbacks.append(
                UploadCheckpointOnTrainEnd(
                    upload_directory=self.cfg.get(
                        "train.callbacks.upload_checkpoint_on_finish.upload_directory"
                    )
                )
            )

        custom_callbacks = self.cfg.get("train.callbacks.custom_callbacks")
        if custom_callbacks:
            for custom_callback in custom_callbacks:
                callbacks.append(
                    CALLBACKS.get(custom_callback.name)(**custom_callback.params)
                )

        return callbacks

    def configure_optimizers(self):
        """Configures model optimizers and schedulers"""
        cfg_optimizers = self.cfg.get("train.optimizers")
        # config params + model parameters
        optim_params = {
            **cfg_optimizers.optimizer.params,
            "params": filter(lambda p: p.requires_grad, self.model.parameters()),
        }
        optimizer = OPTIMIZERS.get(cfg_optimizers.optimizer.name)(**optim_params)

        # config params + optimizer
        scheduler_params = {
            **cfg_optimizers.scheduler.params,
            "optimizer": optimizer,
        }
        scheduler = SCHEDULERS.get(cfg_optimizers.scheduler.name)(**scheduler_params)

        return [optimizer], [scheduler]

    def load_checkpoint(self, path: str):
        """Loads checkpoint weights from provided path"""
        print(f"Loading weights from: {path}")
        fs = LuxonisFileSystem(path)
        checkpoint = torch.load(fs.read_to_byte_buffer())
        state_dict = checkpoint["state_dict"]
        self.load_state_dict(state_dict)

    def forward(self, inputs: torch.Tensor):
        """Calls forward method of the model and returns its output"""
        outputs = self.model(inputs)
        return outputs

    def training_step(self, train_batch: tuple, batch_idx: int):
        """Performs one step of training with provided batch"""
        inputs = train_batch[0]
        label_dict = train_batch[1]

        outputs = self.forward(inputs)

        loss = 0
        sub_losses = []  # list of Tuple(loss_name, value)
        for i, output in enumerate(outputs):
            curr_head = self.model.heads[i]
            curr_head_name = curr_head.get_name(i)
            output_loss, label_loss = curr_head.postprocess_for_loss(output, label_dict)
            curr_loss = self.losses[i](
                output_loss,
                label_loss,
                epoch=self.current_epoch,
                step=self.global_step,
            )
            # if returned loss is tuple
            if isinstance(curr_loss, tuple):
                curr_loss, curr_sub_losses = curr_loss
                # change curr_sub_losses names to be more descriptive and save into joined list
                sub_losses.extend(
                    [(f"{curr_head_name}_{k}", v) for k, v in curr_sub_losses.items()]
                )
            sub_losses.append((curr_head_name, curr_loss.detach()))

            loss += curr_loss * self.cfg.get("train.losses.weights")[i]

            with torch.no_grad():
                if self._is_train_eval_epoch():
                    curr_metrics = self.metrics[curr_head_name]["train_metrics"]
                    (
                        output_metrics,
                        label_metrics,
                        metric_mapping,
                    ) = curr_head.postprocess_for_metric(output, label_dict)
                    if metric_mapping is None:
                        curr_metrics.update(output_metrics, label_metrics)
                    else:
                        for k, v in metric_mapping.items():
                            curr_metrics[k].update(output_metrics[v], label_metrics[v])

                    # images for visualization and logging
                    if batch_idx == 0:
                        unnormalize_img = self.cfg.get(
                            "train.preprocessing.normalize.active"
                        )
                        normalize_params = self.cfg.get(
                            "train.preprocessing.normalize.params"
                        )
                        cvt_color = not self.cfg.get("train.preprocessing.train_rgb")
                        label_imgs = draw_labels(
                            imgs=inputs,
                            label_dict=label_dict,
                            label_keys=curr_head.label_types,
                            unnormalize_img=unnormalize_img,
                            cvt_color=cvt_color,
                            overlay=True,
                            normalize_params=normalize_params,
                        )
                        output_imgs = draw_outputs(
                            imgs=inputs,
                            output=output,
                            head=curr_head,
                            unnormalize_img=unnormalize_img,
                            cvt_color=cvt_color,
                            normalize_params=normalize_params,
                        )
                        merged_imgs = [
                            cv2.hconcat([l_img, o_img])
                            for l_img, o_img in zip(label_imgs, output_imgs)
                        ]

                        num_log_images = self.cfg.get("train.num_log_images")
                        log_imgs = merged_imgs[:num_log_images]

                        for i, img in enumerate(log_imgs):
                            self.logger.log_image(
                                f"train_image/{curr_head_name}_img{i}",
                                img,
                                step=self.current_epoch,
                            )

        step_output = {
            "loss": loss.detach().cpu(),
        }
        if self.cfg.get("train.losses.log_sub_losses"):
            step_output.update({i[0]: i[1].detach().cpu() for i in sub_losses})
        self.training_step_outputs.append(step_output)

        return loss

    def validation_step(self, val_batch: tuple, batch_idx: int):
        """Performs one step of validation with provided batch"""
        inputs = val_batch[0]
        label_dict = val_batch[1]
        outputs = self.forward(inputs)

        loss = 0
        sub_losses = []
        for i, output in enumerate(outputs):
            curr_head = self.model.heads[i]
            curr_head_name = curr_head.get_name(i)
            output_loss, label_loss = curr_head.postprocess_for_loss(output, label_dict)
            curr_loss = self.losses[i](
                output_loss,
                label_loss,
                epoch=self.current_epoch,
                step=self.global_step,
            )
            # if returned loss is tuple
            if isinstance(curr_loss, tuple):
                curr_loss, curr_sub_losses = curr_loss
                # change curr_sub_losses names to be more descriptive and save into joined list
                sub_losses.extend(
                    [(f"{curr_head_name}_{k}", v) for k, v in curr_sub_losses.items()]
                )
            sub_losses.append((curr_head_name, curr_loss.detach()))

            loss += curr_loss * self.cfg.get("train.losses.weights")[i]

            curr_metrics = self.metrics[curr_head_name]["val_metrics"]
            (
                output_metrics,
                label_metrics,
                metric_mapping,
            ) = curr_head.postprocess_for_metric(output, label_dict)
            if metric_mapping is None:
                curr_metrics.update(output_metrics, label_metrics)
            else:
                for k, v in metric_mapping.items():
                    curr_metrics[k].update(output_metrics[v], label_metrics[v])

            # images for visualization and logging
            if batch_idx == 0:
                unnormalize_img = self.cfg.get("train.preprocessing.normalize.active")
                normalize_params = self.cfg.get("train.preprocessing.normalize.params")
                cvt_color = not self.cfg.get("train.preprocessing.train_rgb")
                label_imgs = draw_labels(
                    imgs=inputs,
                    label_dict=label_dict,
                    label_keys=curr_head.label_types,
                    unnormalize_img=unnormalize_img,
                    cvt_color=cvt_color,
                    overlay=True,
                    normalize_params=normalize_params,
                )
                output_imgs = draw_outputs(
                    imgs=inputs,
                    output=output,
                    head=curr_head,
                    unnormalize_img=unnormalize_img,
                    cvt_color=cvt_color,
                    normalize_params=normalize_params,
                )

                merged_imgs = [
                    cv2.hconcat([l_img, o_img])
                    for l_img, o_img in zip(label_imgs, output_imgs)
                ]

                num_log_images = self.cfg.get("train.num_log_images")
                log_imgs = merged_imgs[:num_log_images]

                for i, img in enumerate(log_imgs):
                    self.logger.log_image(
                        f"val_image/{curr_head_name}_img{i}",
                        img,
                        step=self.current_epoch,
                    )

        step_output = {
            "loss": loss.detach().cpu(),
        }
        if self.cfg.get("train.losses.log_sub_losses"):
            step_output.update({i[0]: i[1].detach().cpu() for i in sub_losses})
        self.validation_step_outputs.append(step_output)

        return step_output

    def test_step(self, test_batch: tuple, batch_idx: int):
        """Performs one step of test with provided batch"""
        inputs = test_batch[0]
        label_dict = test_batch[1]
        outputs = self.forward(inputs)

        loss = 0
        sub_losses = []
        for i, output in enumerate(outputs):
            curr_head = self.model.heads[i]
            curr_head_name = curr_head.get_name(i)
            output_loss, label_loss = curr_head.postprocess_for_loss(output, label_dict)
            curr_loss = self.losses[i](
                output_loss, label_loss, epoch=self.current_epoch, step=self.global_step
            )
            # if returned loss is tuple
            if isinstance(curr_loss, tuple):
                curr_loss, curr_sub_losses = curr_loss
                # change curr_sub_losses names to be more descriptive and save into joined list
                sub_losses.extend(
                    [(f"{curr_head_name}_{k}", v) for k, v in curr_sub_losses.items()]
                )
            sub_losses.append((curr_head_name, curr_loss.detach()))

            loss += curr_loss * self.cfg.get("train.losses.weights")[i]

            curr_metrics = self.metrics[curr_head_name]["test_metrics"]
            (
                output_metrics,
                label_metrics,
                metric_mapping,
            ) = curr_head.postprocess_for_metric(output, label_dict)
            if metric_mapping is None:
                curr_metrics.update(output_metrics, label_metrics)
            else:
                for k, v in metric_mapping.items():
                    curr_metrics[k].update(output_metrics[v], label_metrics[v])

            # images for visualization and logging
            if batch_idx == 0:
                unnormalize_img = self.cfg.get("train.preprocessing.normalize.active")
                normalize_params = self.cfg.get("train.preprocessing.normalize.params")
                cvt_color = not self.cfg.get("train.preprocessing.train_rgb")
                label_imgs = draw_labels(
                    imgs=inputs,
                    label_dict=label_dict,
                    label_keys=curr_head.label_types,
                    unnormalize_img=unnormalize_img,
                    cvt_color=cvt_color,
                    overlay=True,
                    normalize_params=normalize_params,
                )
                output_imgs = draw_outputs(
                    imgs=inputs,
                    output=output,
                    head=curr_head,
                    unnormalize_img=unnormalize_img,
                    cvt_color=cvt_color,
                    normalize_params=normalize_params,
                )
                merged_imgs = [
                    cv2.hconcat([l_img, o_img])
                    for l_img, o_img in zip(label_imgs, output_imgs)
                ]

                num_log_images = self.cfg.get("train.num_log_images")
                log_imgs = merged_imgs[:num_log_images]

                for i, img in enumerate(log_imgs):
                    self.logger.log_image(
                        f"test_image/{curr_head_name}_img{i}",
                        img,
                        step=self.current_epoch,
                    )

        step_output = {
            "loss": loss.detach().cpu(),
        }
        if self.cfg.get("train.losses.log_sub_losses"):
            step_output.update({i[0]: i[1].detach().cpu() for i in sub_losses})
        self.test_step_outputs.append(step_output)

        return step_output

    def on_train_epoch_end(self):
        """Performs train epoch end operations"""
        epoch_train_loss = np.mean(
            [step_output["loss"] for step_output in self.training_step_outputs]
        )
        # self.log doesn't have an option to specify step and uses step=self.global_step as default
        # Track this issue if this will change anytime: https://github.com/Lightning-AI/lightning/issues/3228
        self.log("train_loss/loss", epoch_train_loss, sync_dist=True)

        if self.cfg.get("train.losses.log_sub_losses"):
            for key in self.training_step_outputs[0]:
                if key == "loss":
                    continue
                epoch_sub_loss = np.mean(
                    [step_output[key] for step_output in self.training_step_outputs]
                )
                self.log(f"train_loss/{key}", epoch_sub_loss, sync_dist=True)

        metric_results = {}  # used for printing to console
        if self._is_train_eval_epoch():
            self._print_metric_warning("Computing metrics on train subset ...")
            for curr_head_name in self.metrics:
                curr_metrics = self.metrics[curr_head_name]["train_metrics"].compute()
                metric_results[curr_head_name] = curr_metrics
                for metric_name in curr_metrics:
                    self.log(
                        f"train_metric/{curr_head_name}_{metric_name}",
                        curr_metrics[metric_name],
                        sync_dist=True,
                    )
                self.metrics[curr_head_name]["train_metrics"].reset()
            self._print_metric_warning("Metrics computed.")

            if self.cfg.get("trainer.verbose"):
                self._print_results(
                    stage="Train", loss=epoch_train_loss, metrics=metric_results
                )

        self.training_step_outputs.clear()

    def on_validation_epoch_end(self):
        """Performs validation epoch end operations"""
        epoch_val_loss = np.mean(
            [step_output["loss"] for step_output in self.validation_step_outputs]
        )
        self.log("val_loss/loss", epoch_val_loss, sync_dist=True)

        if self.cfg.get("train.losses.log_sub_losses"):
            for key in self.validation_step_outputs[0]:
                if key == "loss":
                    continue
                epoch_sub_loss = np.mean(
                    [step_output[key] for step_output in self.validation_step_outputs]
                )
                self.log(f"val_loss/{key}", epoch_sub_loss, sync_dist=True)

        metric_results = {}  # used for printing to console
        self._print_metric_warning("Computing metrics on val subset ...")
        for i, curr_head_name in enumerate(self.metrics):
            curr_metrics = self.metrics[curr_head_name]["val_metrics"].compute()
            metric_results[curr_head_name] = curr_metrics
            for metric_name in curr_metrics:
                self.log(
                    f"val_metric/{curr_head_name}_{metric_name}",
                    curr_metrics[metric_name],
                    sync_dist=True,
                )
            # log main metrics separately (used in callback)
            if i == 0:
                self.log(
                    f"val_metric/{self.main_metric}",
                    curr_metrics[self.main_metric],
                    sync_dist=True,
                )
            self.metrics[curr_head_name]["val_metrics"].reset()
        self._print_metric_warning("Metrics computed.")

        if self.cfg.get("trainer.verbose"):
            self._print_results(
                stage="Validation", loss=epoch_val_loss, metrics=metric_results
            )

        self.validation_step_outputs.clear()

    def on_test_epoch_end(self):
        """Performs test epoch end operations"""
        epoch_test_loss = np.mean(
            [step_output["loss"] for step_output in self.test_step_outputs]
        )
        self.log("test_loss/loss", epoch_test_loss, sync_dist=True)

        if self.cfg.get("train.losses.log_sub_losses"):
            for key in self.test_step_outputs[0]:
                if key == "loss":
                    continue
                epoch_sub_loss = np.mean(
                    [step_output[key] for step_output in self.test_step_outputs]
                )
                self.log(f"test_loss/{key}", epoch_sub_loss, sync_dist=True)

        metric_results = {}  # used for printing to console
        self._print_metric_warning("Computing metrics on test subset ...")
        for i, curr_head_name in enumerate(self.metrics):
            curr_metrics = self.metrics[curr_head_name]["test_metrics"].compute()
            metric_results[curr_head_name] = curr_metrics
            for metric_name in curr_metrics:
                self.log(
                    f"test_metric/{curr_head_name}_{metric_name}",
                    curr_metrics[metric_name],
                    sync_dist=True,
                )
            # log main metrics separately (used in callback)
            if i == 0:
                self.log(
                    f"test_metric/{self.main_metric}",
                    curr_metrics[self.main_metric],
                    sync_dist=True,
                )
            self.metrics[curr_head_name]["test_metrics"].reset()
        self._print_metric_warning("Metrics computed.")

        if self.cfg.get("trainer.verbose"):
            self._print_results(
                stage="Test", loss=epoch_test_loss, metrics=metric_results
            )

        self.test_step_outputs.clear()

    def get_status(self):
        """Returns current epoch and number of all epochs"""
        return self.current_epoch, self.cfg.get("train.epochs")

    def get_status_percentage(self):
        """Returns percentage of current training, takes into account early stopping"""
        if self._trainer.early_stopping_callback:
            # model didn't yet stop from early stopping callback
            if self._trainer.early_stopping_callback.stopped_epoch == 0:
                return (self.current_epoch / self.cfg.get("train.epochs")) * 100
            else:
                return 100.0
        else:
            return (self.current_epoch / self.cfg.get("train.epochs")) * 100

    def _is_train_eval_epoch(self):
        """Checks if train eval should be performed on current epoch based on configured train_metrics_interval"""
        train_metrics_interval = self.cfg.get("train.train_metrics_interval")
        # add +1 to current_epoch because starting epoch is at 0
        return (
            train_metrics_interval != -1
            and (self.current_epoch + 1) % train_metrics_interval == 0
        )

    def _print_metric_warning(self, text: str):
        """Prints warning in the console for running metric computation (which can take quite long)"""
        if self.cfg.get("train.use_rich_text"):
            # Spinner element would be nicer but afaik needs context manager
            self._trainer.progress_bar_callback.print_single_line(text)
        else:
            print(f"\n{text}")

    @rank_zero_only
    def _print_results(self, stage: str, loss: float, metrics: dict):
        """Prints validation metrics in the console"""
        if self.cfg.get("train.use_rich_text"):
            self._trainer.progress_bar_callback.print_results(
                stage=stage, loss=loss, metrics=metrics
            )
        else:
            print(f"\n----- {stage} -----")
            print(f"Loss: {loss}")
            print(f"Metrics:")
            pprint(metrics)
            print("----------")
