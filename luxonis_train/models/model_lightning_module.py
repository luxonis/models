import pytorch_lightning as pl
import torch
import warnings
import cv2
import torch.nn as nn
import numpy as np
from copy import deepcopy
from pprint import pprint
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.utilities import rank_zero_only

from luxonis_train.models import Model
from luxonis_train.utils.config import Config
from luxonis_train.utils.losses import init_loss
from luxonis_train.utils.optimizers import init_optimizer
from luxonis_train.utils.schedulers import init_scheduler
from luxonis_train.utils.losses import *
from luxonis_train.utils.metrics import init_metrics, postprocess_for_metrics
from luxonis_train.utils.head_type import *
from luxonis_train.utils.general import *
from luxonis_train.utils.visualization import *


class ModelLightningModule(pl.LightningModule):
    def __init__(self, save_dir: str):
        """ Main class used to build and train the model using Pytorch Lightning """
        super().__init__()

        self.cfg = Config()
        self.save_dir = save_dir
        self.model_name = self.cfg.get("model.name")
        self.early_stopping = None # early stopping callback

        self.model = Model()
        self.model.build_model()
        
        # for each head get its loss
        self.losses = nn.ModuleList()
        for head in self.cfg.get("model.heads"):
            if head["loss"]["name"] == "YoloV7PoseLoss":
                for _head in self.model.heads:
                    if isinstance(_head.type, KeyPointDetection):
                        self.losses.append(
                            init_loss(head["loss"]["name"], model=_head)
                        )
                        break
            else:
                self.losses.append(
                    init_loss(head["loss"]["name"], **head["loss"]["params"])
                )

        # for each head initialize its metrics
        self.metrics = nn.ModuleDict()
        for i, curr_head in enumerate(self.model.heads):
            self.metrics[get_head_name(curr_head, i)] = init_metrics(curr_head)

        # load pretrained weights if defined
        if self.cfg.get("model.pretrained"):
            self.load_checkpoint(self.cfg.get("model.pretrained"))

        # freeze modules if defined
        self.freeze_modules(self.cfg.get("train.freeze_modules"))

        # lists for storing intermediate step outputs
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def freeze_modules(self, freeze_info: dict):
        """ Selectively freezes models modules from the training

        Args:
            freeze_info (dict): Dictionary of model parts (backbone|neck|heads) with boolean values set to True if the part should be frozen
        """
        modules_to_freeze = []
        for key, value in freeze_info.items():
            if key == "backbone" and value:
                modules_to_freeze.append(self.model.backbone)
            elif key == "neck" and value:
                if self.model.neck:
                    modules_to_freeze.append(self.model.neck)
                else:
                    warnings.warn("Skipping neck freezing as model doesn't have a neck.")
            elif key == "heads":
                modules_to_freeze.extend([self.model.heads[i] for i, v in enumerate(value) if v])
        
        for module in modules_to_freeze:
            for param in module.parameters():
                param.requires_grad = False

    def configure_callbacks(self):
        """ Configures Pytorch Lightning callbacks """  
        self.min_val_loss_checkpoints_path = f"{self.save_dir}/min_val_loss"
        self.best_val_metric_checkpoints_path = f"{self.save_dir}/best_val_metric"
        self.main_metric = self.model.heads[0].type.main_metric
        
        loss_checkpoint = ModelCheckpoint(
            monitor = "val_loss/loss",
            dirpath = self.min_val_loss_checkpoints_path,
            filename = "loss={val_loss/loss:.4f}_"+self.main_metric+"={val_metric/"+ \
                self.main_metric+":.4f}_{epoch:02d}_" + self.model_name,
            auto_insert_metric_name=False,
            save_top_k = self.cfg.get("train.callbacks.model_checkpoint.save_top_k"),
            mode = "min",
        )

        metric_checkpoint = ModelCheckpoint(
            monitor = f"val_metric/{self.main_metric}",
            dirpath = self.best_val_metric_checkpoints_path,
            filename = self.main_metric+"={val_metric/"+self.main_metric+ \
                ":.4f}_loss={val_loss/loss:.4f}_{epoch:02d}_" + self.model_name,
            auto_insert_metric_name=False,
            save_top_k = self.cfg.get("train.callbacks.model_checkpoint.save_top_k"),
            mode = "max",
        )
        
        lr_monitor = LearningRateMonitor(logging_interval="step")
        callbacks = [loss_checkpoint, metric_checkpoint, lr_monitor]

        # used if we want to perform fine-grained debugging
        if self.cfg.get("train.callbacks.use_device_stats_monitor"):
            from pytorch_lightning.callbacks import DeviceStatsMonitor
            device_stats = DeviceStatsMonitor()
            callbacks.append(device_stats)

        if self.cfg.get("train.callbacks.early_stopping.active"):
            from pytorch_lightning.callbacks import EarlyStopping
            cfg_early_stopping = deepcopy(self.cfg.get("train.callbacks.early_stopping"))
            cfg_early_stopping.pop("active")
            early_stopping = EarlyStopping(**cfg_early_stopping)
            callbacks.append(early_stopping)

        if self.cfg.get("train.use_rich_text"):
            from pytorch_lightning.callbacks import RichModelSummary
            callbacks.append(RichModelSummary())

        return callbacks

    def configure_optimizers(self):
        """ Configures model optimizers and schedulers """
        cfg_optimizer = self.cfg.get("train.optimizers")
        optimizer_name = cfg_optimizer["optimizer"]["name"]
        optimizer = init_optimizer(
            model_params=self.model.parameters(), 
            name=optimizer_name, 
            **cfg_optimizer["optimizer"]["params"] if cfg_optimizer["optimizer"]["params"] else {}
        )
        
        scheduler_name = cfg_optimizer["scheduler"]["name"]
        scheduler = init_scheduler(
            optimizer=optimizer,
            name=scheduler_name,
            **cfg_optimizer["scheduler"]["params"] if cfg_optimizer["scheduler"]["params"] else {}
        )
        return [optimizer], [scheduler]

    def load_checkpoint(self, path: str):
        """ Loads checkpoint weights from provided path """
        print(f"Loading weights from: {path}")
        state_dict = torch.load(path)["state_dict"]
        # remove weights that are not part of the model
        removed = []
        for key in state_dict.keys():
            if not key.startswith("model"):
                removed.append(key)
                state_dict.pop(key)
        if len(removed):
            print(f"Following weights weren't loaded: {removed}")

        self.load_state_dict(state_dict)

    def forward(self, inputs: torch.Tensor):
        """ Calls forward method of the model and returns its output """
        outputs = self.model(inputs)
        return outputs

    def training_step(self, train_batch: tuple, batch_idx: int):
        """ Performs one step of training with provided batch """
        inputs = train_batch[0]
        labels = train_batch[1]
        
        outputs = self.forward(inputs)

        loss = 0
        sub_losses = [] # list of Tuple(loss_name, value)
        for i, output in enumerate(outputs):
            curr_head = self.model.heads[i]
            curr_head_name = get_head_name(curr_head, i)
            curr_label = get_current_label(curr_head.type, labels)
            curr_loss = self.losses[i](
                output, curr_label, 
                epoch=self.current_epoch,
                step=self.global_step,
            )
            # if returned loss is tuple
            if isinstance(curr_loss, tuple):
                curr_loss, curr_sub_losses = curr_loss
                # change curr_sub_losses names to be more descriptive and save into joined list
                sub_losses.extend([(f"{curr_head_name}_{k}",v) for k,v in curr_sub_losses.items()])
            sub_losses.append((curr_head_name, curr_loss.detach()))

            loss += curr_loss * self.cfg.get("train.losses.weights")[i]

            with torch.no_grad(): 
                train_metric_interval = self.cfg.get("train.train_metrics_interval")
                if train_metric_interval != -1 and self.current_epoch % train_metric_interval == 0 and \
                    self.current_epoch != 0:     
                    output_processed, curr_label_processed = postprocess_for_metrics(output, curr_label, curr_head)
                    curr_metrics = self.metrics[curr_head_name]["train_metrics"]
                    curr_metrics.update(output_processed, curr_label_processed)

                    # images for visualization and logging
                    if batch_idx == 0:
                        label_imgs = draw_on_images(inputs, curr_label, curr_head, is_label=True)
                        output_imgs = draw_on_images(inputs, output, curr_head, is_label=False)  
                        merged_imgs = [cv2.hconcat([l_img, o_img]) for l_img, o_img in zip(label_imgs, output_imgs)]
                        
                        num_log_images = self.cfg.get("train.num_log_images")
                        log_imgs = merged_imgs[:num_log_images]

                        for i, img in enumerate(log_imgs):
                            self.logger.log_image(f"train/{curr_head_name}_img{i}", img, step=self.current_epoch)

        step_output = {
            "loss": loss.detach().cpu(),
        }
        if self.cfg.get("train.losses.log_sub_losses"):
            step_output.update({i[0]:i[1].detach().cpu() for i in sub_losses})
        self.training_step_outputs.append(step_output)

        return loss

    def validation_step(self, val_batch: tuple, batch_idx: int):
        """ Performs one step of validation with provided batch """
        inputs = val_batch[0]
        labels = val_batch[1]
        outputs = self.forward(inputs)

        loss = 0
        sub_losses = []
        for i, output in enumerate(outputs):
            curr_head = self.model.heads[i]
            curr_head_name = get_head_name(curr_head, i)
            curr_label = get_current_label(curr_head.type, labels)
            curr_loss = self.losses[i](
                output, curr_label,
                epoch=self.current_epoch,
                step=self.global_step
            )
            # if returned loss is tuple
            if isinstance(curr_loss, tuple):
                curr_loss, curr_sub_losses = curr_loss
                # change curr_sub_losses names to be more descriptive and save into joined list
                sub_losses.extend([(f"{curr_head_name}_{k}",v) for k,v in curr_sub_losses.items()])
            sub_losses.append((curr_head_name, curr_loss.detach()))
            
            loss += curr_loss * self.cfg.get("train.losses.weights")[i]

            output_processed, curr_label_processed = postprocess_for_metrics(output, curr_label, curr_head)
            curr_metrics = self.metrics[curr_head_name]["val_metrics"]
            curr_metrics.update(output_processed, curr_label_processed)
            
            # images for visualization and logging
            if batch_idx == 0:
                label_imgs = draw_on_images(inputs, curr_label, curr_head, is_label=True)
                output_imgs = draw_on_images(inputs, output, curr_head, is_label=False)  
                merged_imgs = [cv2.hconcat([l_img, o_img]) for l_img, o_img in zip(label_imgs, output_imgs)]
                num_log_images = self.cfg.get("train.num_log_images")
                log_imgs = merged_imgs[:num_log_images]
                
                for i, img in enumerate(log_imgs):
                    self.logger.log_image(f"val/{curr_head_name}_img{i}", img, step=self.current_epoch)
        
        step_output = {
            "loss": loss.detach().cpu(),
        }
        if self.cfg.get("train.losses.log_sub_losses"):
            step_output.update({i[0]:i[1].detach().cpu() for i in sub_losses})
        self.validation_step_outputs.append(step_output)
        
        return step_output

    def test_step(self, test_batch: tuple, batch_idx: int):
        inputs = test_batch[0]
        labels = test_batch[1]
        outputs = self.forward(inputs)

        loss = 0
        sub_losses = []
        for i, output in enumerate(outputs):
            curr_head = self.model.heads[i]
            curr_head_name = get_head_name(curr_head, i)
            curr_label = get_current_label(curr_head.type, labels)
            curr_loss = self.losses[i](
                output, curr_label,
                epoch=self.current_epoch,
                step=self.global_step
            )
            # if returned loss is tuple
            if isinstance(curr_loss, tuple):
                curr_loss, curr_sub_losses = curr_loss
                # change curr_sub_losses names to be more descriptive and save into joined list
                sub_losses.extend([(f"{curr_head_name}_{k}",v) for k,v in curr_sub_losses.items()])
            sub_losses.append((curr_head_name, curr_loss.detach()))
            
            loss += curr_loss * self.cfg.get("train.losses.weights")[i]
            
            output_processed, curr_label_processed = postprocess_for_metrics(output, curr_label, curr_head)
            curr_metrics = self.metrics[curr_head_name]["test_metrics"]
            curr_metrics.update(output_processed, curr_label_processed)
        
            # images for visualization and logging
            if batch_idx == 0:
                label_imgs = draw_on_images(inputs, curr_label, curr_head, is_label=True)
                output_imgs = draw_on_images(inputs, output, curr_head, is_label=False)  
                merged_imgs = [cv2.hconcat([l_img, o_img]) for l_img, o_img in zip(label_imgs, output_imgs)]
                
                num_log_images = self.cfg.get("train.num_log_images")
                log_imgs = merged_imgs[:num_log_images]

                for i, img in enumerate(log_imgs):
                    self.logger.log_image(f"test/{curr_head_name}_img{i}", img, step=self.current_epoch)

        step_output = {
            "loss": loss.detach().cpu(),
        }
        if self.cfg.get("train.losses.log_sub_losses"):
            step_output.update({i[0]:i[1].detach().cpu() for i in sub_losses})
        self.test_step_outputs.append(step_output)
        
        return step_output

    def on_train_epoch_end(self):
        """ Performs train epoch end operations """
        epoch_train_loss = np.mean([step_output["loss"].item() for step_output in self.training_step_outputs])
        self.log("train_loss/loss", epoch_train_loss, sync_dist=True)

        if self.cfg.get("train.losses.log_sub_losses"):
            for key in self.training_step_outputs[0]:
                if key == "loss":
                    continue
                epoch_sub_loss = np.mean([step_output[key].item() for step_output in self.training_step_outputs])
                self.log(f"train_loss/{key}", epoch_sub_loss, sync_dist=True)

        metric_results = {} # used for printing to console
        train_metric_interval = self.cfg.get("train.train_metrics_interval")
        if train_metric_interval != -1 and self.current_epoch % train_metric_interval == 0 and \
            self.current_epoch != 0:    
            for curr_head_name in self.metrics:
                curr_metrics = self.metrics[curr_head_name]["train_metrics"].compute()
                metric_results[curr_head_name] = curr_metrics
                for metric_name in curr_metrics:
                    self.log(f"train_metric/{curr_head_name}_{metric_name}", curr_metrics[metric_name], sync_dist=True)
                self.metrics[curr_head_name]["train_metrics"].reset() 

            if self.cfg.get("trainer.verbose"):
                self._print_results(stage="Train", loss=epoch_train_loss, metrics=metric_results)

        self.training_step_outputs.clear()

    def on_validation_epoch_end(self):
        """ Performs validation epoch end operations """
        epoch_val_loss = np.mean([step_output["loss"].item() for step_output in self.validation_step_outputs])
        self.log("val_loss/loss", epoch_val_loss, sync_dist=True)
        
        if self.cfg.get("train.losses.log_sub_losses"):
            for key in self.validation_step_outputs[0]:
                if key == "loss":
                    continue
                epoch_sub_loss = np.mean([step_output[key].item() for step_output in self.validation_step_outputs])
                self.log(f"val_loss/{key}", epoch_sub_loss, sync_dist=True)

        metric_results = {} # used for printing to console
        for i, curr_head_name in enumerate(self.metrics):
            curr_metrics = self.metrics[curr_head_name]["val_metrics"].compute()
            metric_results[curr_head_name] = curr_metrics
            for metric_name in curr_metrics:
                self.log(f"val_metric/{curr_head_name}_{metric_name}", curr_metrics[metric_name], sync_dist=True)
            # log main metrics separately (used in callback)
            if i == 0:
                self.log(f"val_metric/{self.main_metric}", curr_metrics[self.main_metric], sync_dist=True)
            self.metrics[curr_head_name]["val_metrics"].reset()

        if self.cfg.get("trainer.verbose"):
            self._print_results(stage="Validation", loss=epoch_val_loss, metrics=metric_results)
        
        self.validation_step_outputs.clear()

    def on_test_epoch_end(self):
        epoch_test_loss = np.mean([step_output["loss"].item() for step_output in self.test_step_outputs])
        self.log("test_loss/loss", epoch_test_loss, sync_dist=True)

        if self.cfg.get("train.losses.log_sub_losses"):
            for key in self.test_step_outputs[0]:
                if key == "loss":
                    continue
                epoch_sub_loss = np.mean([step_output[key].item() for step_output in self.test_step_outputs])
                self.log(f"test_loss/{key}", epoch_sub_loss, sync_dist=True)

        metric_results = {} # used for printing to console
        for i, curr_head_name in enumerate(self.metrics):
            curr_metrics = self.metrics[curr_head_name]["test_metrics"].compute()
            metric_results[curr_head_name] = curr_metrics
            for metric_name in curr_metrics:
                self.log(f"test_metric/{curr_head_name}_{metric_name}", curr_metrics[metric_name], sync_dist=True)
            # log main metrics separately (used in callback)
            if i == 0:
                self.log(f"test_metric/{self.main_metric}", curr_metrics[self.main_metric], sync_dist=True)
            self.metrics[curr_head_name]["test_metrics"].reset()

        if self.cfg.get("trainer.verbose"):
            self._print_results(stage="Test", loss=epoch_test_loss, metrics=metric_results)

        self.test_step_outputs.clear()
    
    def get_status(self):
        """ Returns current epoch and number of all epochs """
        return self.current_epoch, self.cfg.get("train.epochs")
    
    def get_status_percentage(self):
        """ Returns percentage of current training, takes into account early stopping """
        if self._trainer.early_stopping_callback:
             # model didn't yet stop from early stopping callback
            if self._trainer.early_stopping_callback.stopped_epoch == 0:
                return (self.current_epoch / self.cfg.get("train.epochs"))*100
            else:
                return 100.0
        else:    
            return (self.current_epoch / self.cfg.get("train.epochs"))*100 

    @rank_zero_only
    def _print_results(self, stage: str, loss: float, metrics: dict):
        """ Prints validation metrics in the console """
        if self.cfg.get("train.use_rich_text"):
            self._trainer.progress_bar_callback.print_results(stage=stage, loss=loss, metrics=metrics)
        else:
            print(f"\n----- {stage} -----")
            print(f"Loss: {loss}")
            print(f"Metrics:")
            pprint(metrics)
            print("----------")
