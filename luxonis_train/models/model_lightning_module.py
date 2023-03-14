import pytorch_lightning as pl
import torch
import torch.nn as nn
import glob
import warnings
from pprint import pprint
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.utilities import rank_zero_only

from .model import Model
from luxonis_train.utils.config import *
from luxonis_train.utils.losses import get_loss
from luxonis_train.utils.optimizers import get_optimizer
from luxonis_train.utils.schedulers import get_scheduler
from luxonis_train.utils.losses import *
from luxonis_train.utils.metrics import init_metrics, postprocess_for_metrics
from luxonis_train.utils.head_type import *
from luxonis_train.utils.general import *


class ModelLightningModule(pl.LightningModule):
    def __init__(self, cfg, save_dir):
        super().__init__()

        self.cfg = cfg
        self.save_dir = save_dir
        self.model_name = cfg["model"]["name"]

        # check if model is predefined
        if self.cfg["model"]["type"]:
            load_predefined_cfg(self.cfg)

        check_cfg(self.cfg)
        self.model = Model()
        self.model.build_model(self.cfg["model"], self.cfg["train"]["image_size"])

        # for each head get its loss
        self.losses = nn.ModuleList()
        for head in self.cfg["model"]["heads"]:
            self.losses.append(get_loss(head["loss"]["name"], **head["loss"]["params"] if head["loss"]["params"] else {}))

        # for each head initialize its metrics
        self.metrics = nn.ModuleDict()
        for i, curr_head in enumerate(self.model.heads):
            self.metrics[get_head_name(curr_head, i)] = init_metrics(curr_head)

        # load pretrained weights if defined
        if self.cfg["model"]["pretrained"]:
            self.load_checkpoint(self.cfg["model"]["pretrained"])

        # freeze modules if defined
        if "freeze_modules" in self.cfg["train"] and self.cfg["train"]["freeze_modules"]:
            self.freeze_modules(self.cfg["train"]["freeze_modules"])


    def freeze_modules(self, freeze_info):
        modules_to_freeze = []
        for key, value in freeze_info.items():
            if key == "backbone" and value:
                modules_to_freeze.append(self.model.backbone)
            elif key == "neck" and value:
                if self.model.neck:
                    modules_to_freeze.append(self.model.neck)
                else:
                    warnings.warn("Skipping neck freezing as model doesn't have a neck.", SyntaxWarning)
            elif key == "heads":
                if len(value) != len(self.model.heads):
                    raise RuntimeError("Number of heads must match length of 'heads' list under 'freeze_modules'.")
                for i, v in enumerate(value):
                    if v: modules_to_freeze.append(self.model.heads[i])
        for module in modules_to_freeze:
            for param in module.parameters():
                param.requires_grad = False

    def configure_callbacks(self):    
        self.min_val_loss_checkpoints_path = f"{self.save_dir}/min_val_loss"
        self.best_val_metric_checkpoints_path = f"{self.save_dir}/best_val_metric"
        self.main_metric = self.model.heads[0].type.main_metric
        
        loss_checkpoint = ModelCheckpoint(
            monitor = "val_loss",
            dirpath = self.min_val_loss_checkpoints_path,
            filename = "{val_loss:.4f}_{val_"+self.main_metric+":.4f}_{epoch:02d}_" + self.model_name,
            save_top_k = 3,
            mode = "min"
        )

        metric_checkpoint = ModelCheckpoint(
            monitor = f"val_{self.main_metric}",
            dirpath = self.best_val_metric_checkpoints_path,
            filename = "{val_"+self.main_metric+":.4f}_{val_loss:.4f}_{epoch:02d}_" + self.model_name,
            save_top_k = 3,
            mode = "max"
        )
        
        lr_monitor = LearningRateMonitor(logging_interval="step")

        callbacks = [loss_checkpoint, metric_checkpoint, lr_monitor]

        if "early_stopping" in self.cfg["train"]:
            callbacks.append(EarlyStopping(**self.cfg["train"]["early_stopping"]))

        return callbacks

    def load_best_checkpoint(self):
        checkpoint_paths = glob.glob(
            f"{self.min_val_loss_checkpoints_path}/*"
        )
        # lowest val_loss
        best_checkpoint_path = sorted(checkpoint_paths)[0]
        lowest_val_loss = best_checkpoint_path.split("/")[-1].split("_")[1].replace("loss=", "")
        self.load_state_dict(
            torch.load(best_checkpoint_path)["state_dict"]
        )
        return lowest_val_loss

    def load_checkpoint(self, path):
        print(f"Loading model weights from: {path}")
        self.load_state_dict(torch.load(path)["state_dict"])

    def export(self):
        pass

    def forward(self, inputs):
        outputs = self.model(inputs)
        return outputs

    def configure_optimizers(self):
        optimizer_name = self.cfg["optimizer"]["name"]
        optimizer = get_optimizer(
            model_params=self.model.parameters(), 
            name=optimizer_name, 
            **self.cfg["optimizer"]["params"] if self.cfg["optimizer"]["params"] else {}
        )
        
        scheduler_name = self.cfg["scheduler"]["name"]
        scheduler = get_scheduler(
            optimizer=optimizer,
            name=scheduler_name,
            **self.cfg["scheduler"]["params"] if self.cfg["scheduler"]["params"] else {}
        )
        return [optimizer], [scheduler]

    def training_step(self, train_batch, batch_idx):
        inputs = train_batch[0].float()
        labels = train_batch[1]
        outputs = self.forward(inputs)

        loss = 0
        for i, output in enumerate(outputs):
            curr_head = self.model.heads[i]
            curr_head_name = get_head_name(curr_head, i)
            curr_label = get_current_label(curr_head.type, labels)
            curr_loss = self.losses[i](output, curr_label, epoch=self.current_epoch,
                step=self.global_step, original_in_shape=self.cfg["train"]["image_size"])
            loss += curr_loss

            with torch.no_grad():         
                if self.cfg["train"]["n_metrics"] and self.current_epoch % self.cfg["train"]["n_metrics"] == 0:
                    output_processed, curr_label_processed = postprocess_for_metrics(output, curr_label, curr_head)
                    curr_metrics = self.metrics[curr_head_name]["train_metrics"]
                    curr_metrics.update(output_processed, curr_label_processed)

        # loss required in step output by pl
        step_output = {
            "loss": loss
        }
        return step_output

    def validation_step(self, val_batch, batch_idx):
        inputs = val_batch[0].float()
        labels = val_batch[1]
        outputs = self.forward(inputs)

        loss = 0
        for i, output in enumerate(outputs):
            curr_head = self.model.heads[i]
            curr_head_name = get_head_name(curr_head, i)
            curr_label = get_current_label(curr_head.type, labels)
            curr_loss = self.losses[i](output, curr_label, epoch=self.current_epoch,
                step=self.global_step, original_in_shape=self.cfg["train"]["image_size"])
            loss += curr_loss

            output_processed, curr_label_processed = postprocess_for_metrics(output, curr_label, curr_head)
            curr_metrics = self.metrics[curr_head_name]["val_metrics"]
            curr_metrics.update(output_processed, curr_label_processed)
        
        step_output = {
            "loss": loss,
        }
        return step_output
    
    def test_step(self, test_batch, batch_idx):
        inputs = test_batch[0].float()
        labels = test_batch[1]
        outputs = self.forward(inputs)

        loss = 0
        for i, output in enumerate(outputs):
            curr_head = self.model.heads[i]
            curr_head_name = get_head_name(curr_head, i)
            curr_label = get_current_label(curr_head.type, labels)
            curr_loss = self.losses[i](output, curr_label, epoch=self.current_epoch,
                step=self.global_step, original_in_shape=self.cfg["train"]["image_size"])
            loss += curr_loss
            
            output_processed, curr_label_processed = postprocess_for_metrics(output, curr_label, curr_head)
            curr_metrics = self.metrics[curr_head_name]["test_metrics"]
            curr_metrics.update(output_processed, curr_label_processed)
        
        step_output = {
            "loss": loss,
        }
        return step_output

    def training_epoch_end(self, outputs):
        epoch_train_loss = self._avg([step_output["loss"] for step_output in outputs])
        self.log("train_loss", epoch_train_loss, sync_dist=True)

        if self.cfg["train"]["n_metrics"] and self.current_epoch % self.cfg["train"]["n_metrics"] == 0:
            for curr_head_name in self.metrics:
                curr_metrics = self.metrics[curr_head_name]["train_metrics"].compute()
                for metric_name in curr_metrics:
                    self.log(f"{curr_head_name}_{metric_name}/train", curr_metrics[metric_name], sync_dist=True)
                self.metrics[curr_head_name]["train_metrics"].reset() 

    def validation_epoch_end(self, outputs):
        epoch_val_loss = self._avg([step_output["loss"] for step_output in outputs])
        self.log("val_loss", epoch_val_loss, sync_dist=True)
        
        results = {} # used for printing to console
        for i, curr_head_name in enumerate(self.metrics):
            curr_metrics = self.metrics[curr_head_name]["val_metrics"].compute()
            results[curr_head_name] = curr_metrics
            for metric_name in curr_metrics:
                self.log(f"{curr_head_name}_{metric_name}/val", curr_metrics[metric_name], sync_dist=True)
            # log main metrics separately (used in callback)
            if i == 0:
                self.log(f"val_{self.main_metric}", curr_metrics[self.main_metric], sync_dist=True)
            self.metrics[curr_head_name]["val_metrics"].reset()

        self._print_results(epoch_val_loss, results)

    def test_epoch_end(self, outputs):
        # TODO: what do we want to log/show on test epoch end? same as validation epoch end?
        pass
    
    def get_status(self):
        # return current epoch and number of all epochs
        return self.current_epoch, self.cfg["train"]["epochs"]

    def _avg(self, running_metric):
        return sum(running_metric) / len(running_metric)

    @rank_zero_only
    def _print_results(self, loss, metrics):
        print("Validation metrics:")
        print(f"Val_loss: {loss}")
        pprint(metrics)  