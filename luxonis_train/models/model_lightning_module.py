import pytorch_lightning as pl
import torch
import torch.nn as nn
import glob
import warnings
from pprint import pprint
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, DeviceStatsMonitor
from pytorch_lightning.utilities import rank_zero_only
from torchvision.utils import draw_bounding_boxes

from .model import Model
from luxonis_train.utils.losses import get_loss
from luxonis_train.utils.optimizers import get_optimizer
from luxonis_train.utils.schedulers import get_scheduler
from luxonis_train.utils.losses import *
from luxonis_train.utils.metrics import init_metrics, postprocess_for_metrics
from luxonis_train.utils.head_type import *
from luxonis_train.utils.general import *

from luxonis_train.utils.visualization import *

import torch.nn.functional as F
from luxonis_train.utils.head_utils import yolov6_out2box


class ModelLightningModule(pl.LightningModule):
    def __init__(self, cfg, save_dir):
        super().__init__()

        self.cfg = cfg
        self.save_dir = save_dir
        self.model_name = cfg["model"]["name"]
        self.early_stopping = None # early stopping callback

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
        
        if "model_checkpoint" in self.cfg["train"] and "save_top_k" in self.cfg["train"]["model_checkpoint"] and self.cfg["train"]["model_checkpoint"]["save_top_k"]:
            self.save_top_k = self.cfg["train"]["model_checkpoint"]["save_top_k"]
        else:
            self.save_top_k = 3

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
            save_top_k = self.save_top_k,
            mode = "min"
        )

        metric_checkpoint = ModelCheckpoint(
            monitor = f"val_{self.main_metric}",
            dirpath = self.best_val_metric_checkpoints_path,
            filename = "{val_"+self.main_metric+":.4f}_{val_loss:.4f}_{epoch:02d}_" + self.model_name,
            save_top_k = self.save_top_k,
            mode = "max"
        )
        
        lr_monitor = LearningRateMonitor(logging_interval="step")
        callbacks = [loss_checkpoint, metric_checkpoint, lr_monitor]

        # used if we want to perform fine-grained debugging
        # device_stats = DeviceStatsMonitor()
        # callbacks.append(device_stats)

        if "early_stopping" in self.cfg["train"]:
            self.early_stopping = EarlyStopping(verbose=True, **self.cfg["train"]["early_stopping"])
            callbacks.append(self.early_stopping)

        return callbacks

    def load_checkpoint(self, path):
        print(f"Loading model weights from: {path}")
        self.load_state_dict(torch.load(path)["state_dict"])

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
                if self.cfg["train"]["n_metrics"] and self.current_epoch % self.cfg["train"]["n_metrics"] == 0 and self.current_epoch != 0:
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
            
            if batch_idx == 0: # log images of the first batch
                if isinstance(curr_head.type, Classification):
                    labels = torch.argmax(curr_label, dim=1)
                    predictions = torch.argmax(output, dim=1)
                    # TODO: also read-in and draw on images class names instead of indexes
                    images_to_log = []
                    for i, (img, label, prediction) in enumerate(zip(inputs, labels, predictions)):
                        label = int(label)
                        prediction = int(prediction)
                        img = unnormalize(img, to_uint8=True)
                        img = torch_to_cv2(img, to_rgb=True)
                        img_data = {"label":f"{label}", "prediction":f"{prediction}"}
                        images_to_log.append(draw_on_image(img, img_data, curr_head))
                        if i>=2: # only log a few images
                            break

                    ## logging
                    concatenate = images_to_log.pop(0)
                    while images_to_log != []:
                        concatenate = np.concatenate((concatenate, images_to_log.pop(0)), axis=1)
                    self.logger.log_image(curr_head_name, concatenate, step=self.current_epoch)
                    # could also use:
                    #self.logger.log_images(curr_head_name, np.array(images_to_log), step=self.current_epoch)

                    
                elif isinstance(curr_head.type, ObjectDetection):
                    
                    images_to_log = []

                    output_nms = yolov6_out2box(output, curr_head) # TODO: add conf_thres= and iout_thres arguments as e.g. yolov6_out2box(output, curr_head, conf_thres=0.3, iout_thres=0.6)

                    for i, img in enumerate(inputs):
                        
                        img = unnormalize(img, to_uint8=True)
                        
                        ## labeled bboxes
                        bboxes = None
                        for bbox in labels["bbox"]:
                            if bbox[0] == i: ## check if bbox belongs to img
                                if bboxes == None:
                                    bboxes = bbox.unsqueeze(0)
                                else:
                                    bboxes = torch.cat((bboxes, bbox.unsqueeze(0)), 0)

                        img1 = draw_on_image(img, bboxes, curr_head, is_label=True) if bboxes != None else img.detach().clone()
                        img1 = torch_to_cv2(img1, to_rgb=True)

                        ## predicted bboxes
                        pred_bboxs = output_nms[i][:,:4]
                        pred_labels = output_nms[i][:,5].int()
                        ##look into visualisations.py draw on image to also draw labels!
                        img2 = draw_bounding_boxes(img, pred_bboxs, labels=None)
                        img2 = torch_to_cv2(img2, to_rgb=True)
                        
                        images_to_log.append(np.concatenate((img1, img2), axis=0))
                        
                        if i>=2: # only log a few images
                            break

                    ## logging
                    concatenate = images_to_log.pop(0)
                    while images_to_log != []:
                        concatenate = np.concatenate((concatenate, images_to_log.pop(0)), axis=1)
                    self.logger.log_image(curr_head_name, concatenate, step=self.current_epoch)
        
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
                    self.log(f"train/{curr_head_name}_{metric_name}", curr_metrics[metric_name], sync_dist=True)
                self.metrics[curr_head_name]["train_metrics"].reset() 

    def validation_epoch_end(self, outputs):
        epoch_val_loss = self._avg([step_output["loss"] for step_output in outputs])
        self.log("val_loss", epoch_val_loss, sync_dist=True)
        
        results = {} # used for printing to console
        for i, curr_head_name in enumerate(self.metrics):
            curr_metrics = self.metrics[curr_head_name]["val_metrics"].compute()
            results[curr_head_name] = curr_metrics
            for metric_name in curr_metrics:
                self.log(f"val/{curr_head_name}_{metric_name}", curr_metrics[metric_name], sync_dist=True)
            # log main metrics separately (used in callback)
            if i == 0:
                self.log(f"val_{self.main_metric}", curr_metrics[self.main_metric], sync_dist=True)
            self.metrics[curr_head_name]["val_metrics"].reset()

        self._print_results(epoch_val_loss, results)

    def test_epoch_end(self, outputs):
        epoch_test_loss = self._avg([step_output["loss"] for step_output in outputs])
        self.log("test_loss", epoch_test_loss, sync_dist=True)

        results = {} # used for printing to console
        for i, curr_head_name in enumerate(self.metrics):
            curr_metrics = self.metrics[curr_head_name]["test_metrics"].compute()
            results[curr_head_name] = curr_metrics
            for metric_name in curr_metrics:
                self.log(f"test/{curr_head_name}_{metric_name}", curr_metrics[metric_name], sync_dist=True)
            # log main metrics separately (used in callback)
            if i == 0:
                self.log(f"test_{self.main_metric}", curr_metrics[self.main_metric], sync_dist=True)
            self.metrics[curr_head_name]["test_metrics"].reset()

        self._print_results(epoch_test_loss, results)
    
    def get_status(self):
        """ Return current epoch and number of all epochs """
        return self.current_epoch, self.cfg["train"]["epochs"]
    
    def get_status_percentage(self):
        """ Return percentage of current training, takes into account early stopping """
        if self.early_stopping:
             # model didn't yet stop from early stopping callback
            if self.early_stopping.stopped_epoch == 0:
                return (self.current_epoch / self.cfg["train"]["epochs"])*100
            else:
                return 100.0
        else:    
            return (self.current_epoch / self.cfg["train"]["epochs"])*100

    def get_n_classes(self):
        """ Return n_classes for each type of annotation """
        out_dict = {}
        for head in self.model.heads:
            if isinstance(head.type, Classification) or isinstance(head.type, MultiLabelClassification):
                out_dict["class"] = head.n_classes
            elif isinstance(head.type, SemanticSegmentation) or isinstance(head.type, InstanceSegmentation):
                out_dict["segmentation"] = head.n_classes
            # TODO: do we need the same for object detection and keypoint detection?
        return out_dict            

    def _avg(self, running_metric):
        return sum(running_metric) / len(running_metric)

    @rank_zero_only
    def _print_results(self, loss, metrics):
        print("\nValidation metrics:")
        print(f"Val_loss: {loss}")
        pprint(metrics)  
