from collections import OrderedDict, defaultdict
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from functools import partial
from pprint import pformat, pprint
from typing import Any, cast

import numpy as np
import pytorch_lightning as pl
import torch
from luxonis_ml.data import LuxonisDataset
from luxonis_ml.loader import LuxonisLoader, TrainAugmentations
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.utilities import rank_zero_only  # type: ignore
from rich import print
from torch import Tensor, nn

from luxonis_train.losses import LuxonisLoss
from luxonis_train.metrics.luxonis_metric import LuxonisMetric
from luxonis_train.models.modules.luxonis_module import LuxonisModule
from luxonis_train.utils.callbacks import ModuleFreezer
from luxonis_train.utils.config import Config
from luxonis_train.utils.filesystem import LuxonisFileSystem
from luxonis_train.utils.registry import (
    CALLBACKS,
    LOSSES,
    METRICS,
    MODULES,
    OPTIMIZERS,
    SCHEDULERS,
    VISUALIZERS,
)
from luxonis_train.utils.types import Labels, ModulePacket, ShapePacket
from luxonis_train.visualizers import LuxonisVisualizer, preprocess_image

Kwargs = dict[str, Any]
NodeName = str
Size = list[int]


@dataclass
class LuxonisOutput:
    outputs: dict[NodeName, ModulePacket]
    losses: dict[NodeName, Tensor]
    visualizations: dict[NodeName, dict[str, Tensor]] = field(default_factory=dict)
    metrics: dict[NodeName, dict[str, Tensor]] = field(default_factory=dict)

    def __str__(self) -> str:
        outputs = {
            k: [list(tensor.shape) for tensor in v] for k, v in self.outputs.items()
        }
        viz = {
            f"{node_name}.{viz_name}": viz_value.shape
            for node_name, viz in self.visualizations.items()
            for viz_name, viz_value in viz.items()
        }
        string = pformat(
            {"outputs": outputs, "visualizations": viz, "losses": self.losses}
        )
        return f"{self.__class__.__name__}(\n{string}\n)"

    def __repr__(self) -> str:
        return str(self)


class LuxonisModel(pl.LightningModule):
    def __init__(
        self,
        save_dir: str,
        nodes: dict[NodeName, tuple[type[LuxonisModule], Kwargs]],
        graph: OrderedDict[NodeName, list[NodeName]],
        losses: dict[NodeName, tuple[type[LuxonisLoss], Kwargs]],
        loss_weights: dict[NodeName, float],
        input_shapes: dict[NodeName, list[Size]],
        outputs: list[NodeName],
        visualizers: dict[NodeName, list[LuxonisVisualizer]] | None = None,
        metrics: dict[NodeName, list[LuxonisMetric]] | None = None,
    ):
        super().__init__()
        self.cfg = Config()
        self.save_dir = save_dir
        self.model_name = self.cfg.get("model.name")
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

        self.input_shapes = input_shapes
        self.graph = graph
        self.reversed_graph = self._reverse_graph(graph)
        for output_name in outputs:
            self.graph[output_name] = []
        for input_name in input_shapes:
            self.reversed_graph[input_name] = []
        # self.losses = losses
        self.visualizers = nn.ModuleDict(
            {k: nn.ModuleList(v) for k, v in (visualizers or {}).items()}
        )
        self.metrics = nn.ModuleDict(
            {k: nn.ModuleList(v) for k, v in (metrics or {}).items()}
        )

        self.outputs = outputs
        self.eval()
        self.nodes, self.losses = self._initiate_nodes(nodes, losses)
        self.loss_weights = loss_weights

        if self.cfg.get("model").get("pretrained"):
            self.load_checkpoint(self.cfg.get("model.pretrained"))

    @classmethod
    def from_components(
        cls,
        save_dir: str,
        backbone: tuple[type[LuxonisModule], Kwargs],
        neck: tuple[type[LuxonisModule], Kwargs] | None,
        heads: dict[str, tuple[type[LuxonisModule], Kwargs]],
        losses: dict[str, tuple[type[LuxonisLoss], Kwargs]],
        visualizers: dict[str, list[LuxonisVisualizer]],
        loss_weights: dict[str, float],
        metrics: dict[str, list[LuxonisMetric]],
        input_shape: Size,
    ):
        nodes = {"backbone": backbone, **heads}
        if neck is not None:
            nodes["neck"] = neck
            graph = OrderedDict(
                {
                    "backbone": ["neck"],
                    "neck": list(heads.keys()),
                }
            )
        else:
            graph = OrderedDict({"backbone": list(heads.keys())})
        return cls(
            save_dir=save_dir,
            nodes=nodes,
            graph=graph,
            losses=losses,
            loss_weights=loss_weights,
            input_shapes={"backbone": [input_shape]},
            outputs=list(heads.keys()),
            metrics=metrics,
            visualizers=visualizers,
        )

    @classmethod
    def from_config(cls, save_dir: str, cfg: Config):
        backbone = MODULES.get(cfg.get("model.backbone.name"))
        neck = (
            MODULES.get(cfg.get("model.neck.name")) if cfg.get("model.neck") else None
        )
        losses = {}
        loss_weights = {}
        heads = {}
        visualizers = defaultdict(list)
        metrics = defaultdict(list)
        for head in cfg.get("model.heads"):
            head_name = head["name"]
            heads[head_name] = (MODULES.get(head_name), head["params"])

            loss_name = head["loss"]["name"]
            loss_weights[loss_name] = head["loss"].get("weight", 1.0)
            loss_params = head["loss"].get("params") or {}
            losses[head_name] = LOSSES.get(loss_name), loss_params
            for metric in head.get("metrics") or []:
                metric_name = metric["name"]
                metric_params = metric.get("params") or {}
                metrics[head_name].append(METRICS.get(metric_name)(**metric_params))

            for visualizer in head.get("visualizers") or []:
                visualizer_name = visualizer["name"]
                visualizer_params = visualizer.get("params") or {}
                visualizers[head_name].append(
                    VISUALIZERS.get(visualizer_name)(**visualizer_params)
                )

        input_shape = cls._get_input_shape(cfg)
        return cls.from_components(
            save_dir=save_dir,
            backbone=(backbone, cfg.get("model.backbone.params")),
            neck=(neck, cfg.get("model.neck.params")) if neck else None,
            heads=heads,
            losses=losses,
            loss_weights=loss_weights,
            metrics=metrics,
            input_shape=input_shape,
            visualizers=visualizers,
        )

    @staticmethod
    def _get_input_shape(cfg):
        with LuxonisDataset(
            cfg.get("dataset.dataset_name"),
        ) as dataset:
            train_augmentations = TrainAugmentations(
                image_size=cfg.get("train.preprocessing.train_image_size"),
                augmentations=cfg.get("train.preprocessing.augmentations"),
                train_rgb=cfg.get("train.preprocessing.train_rgb"),
                keep_aspect_ratio=cfg.get("train.preprocessing.keep_aspect_ratio"),
            )

            loader_train = LuxonisLoader(
                dataset,
                view=cfg.get("dataset.train_view"),
                augmentations=train_augmentations,
                mode="json" if cfg.get("dataset.json_mode") else "fiftyone",
            )
            img, _ = next(iter(loader_train))
            return [1, *list(img.shape)]

    @staticmethod
    def _reverse_graph(graph):
        reversed_graph = {}

        for node, neighbors in graph.items():
            for neighbor in neighbors:
                if neighbor not in reversed_graph:
                    reversed_graph[neighbor] = []
                reversed_graph[neighbor].append(node)

        return reversed_graph

    def _initiate_nodes(
        self,
        nodes: dict[NodeName, tuple[type[LuxonisModule], Kwargs]],
        losses: dict[NodeName, tuple[type[LuxonisLoss], Kwargs]],
    ) -> tuple[nn.ModuleDict, nn.ModuleDict]:
        """
        Initiate all nodes in the graph. Starts by feeding dummy input to the
        input nodes and using the output shapes to initiate the next nodes.
        Then feeds dummy input to the next nodes and so on.
        """
        # TODO: validate losses/metrics/visualizers
        initiated: dict[NodeName, tuple[LuxonisModule, ModulePacket]] = {}
        initiated_losses: dict[NodeName, LuxonisLoss] = {}

        # Process the input nodes first
        # TODO: remove this duplicity of processing input nodes both here and in forward
        for name, shapes in self.input_shapes.items():
            module_type, kwargs = nodes[name]
            module_instance = module_type(
                input_shapes=[{"features": shapes}], **kwargs
            ).eval()
            dummy_inputs: list[ModulePacket] = [
                {"features": [torch.zeros(shape)]} for shape in shapes
            ]
            dummy_output: ModulePacket = module_instance(dummy_inputs)
            initiated[name] = (module_instance, dummy_output)

        # Process the remaining nodes
        while len(initiated) < len(nodes):
            for node_name, (module_type, kwargs) in nodes.items():
                if node_name in initiated:
                    continue

                if all(pred in initiated for pred in self.reversed_graph[node_name]):
                    dummy_inputs = [
                        initiated[pred][1] for pred in self.reversed_graph[node_name]
                    ]
                    input_shapes: list[ShapePacket] = [
                        {
                            name: [list(tensor.shape) for tensor in tensors]
                            for name, tensors in dummy_input.items()
                        }
                        for dummy_input in dummy_inputs
                    ]
                    module_instance = module_type(
                        input_shapes=input_shapes, **kwargs
                    ).eval()
                    dummy_outputs = module_instance(dummy_inputs)
                    initiated[node_name] = (module_instance, dummy_outputs)
                    if node_name in losses:
                        loss_type, loss_kwargs = losses[node_name]
                        loss = loss_type(
                            module_attributes=module_instance.__dict__, **loss_kwargs
                        ).eval()
                        initiated_losses[node_name] = loss

        return (
            nn.ModuleDict({name: module for name, (module, _) in initiated.items()}),
            nn.ModuleDict(initiated_losses),
        )

    def forward(
        self,
        inputs: dict[NodeName, list[Tensor]],
        labels: Labels | None = None,
        image: Tensor | None = None,
        _cleanup: bool = True,
    ) -> LuxonisOutput:
        computed: dict[NodeName, ModulePacket] = {}
        losses: dict[NodeName, Tensor] = {}
        visualizations: dict[NodeName, dict[str, Tensor]] = defaultdict(dict)
        for name, input_tensors in inputs.items():
            computed[name] = self.nodes[name]([{"features": input_tensors}])

        while not all(output in computed for output in self.outputs):
            for node_name, module in self.nodes.items():
                module = cast(LuxonisModule, module)
                if node_name in computed:
                    continue

                if all(pred in computed for pred in self.reversed_graph[node_name]):
                    module_inputs = [
                        computed[pred] for pred in self.reversed_graph[node_name]
                    ]
                    outputs = module(module_inputs)
                    computed[node_name] = outputs
                    if node_name in self.losses and labels is not None:
                        losses[node_name] = self.losses[node_name].forward(
                            outputs, labels
                        )
                    if (
                        not self.training
                        and node_name in self.metrics
                        and labels is not None
                    ):
                        for metric in self.metrics[node_name]:  # type: ignore
                            preds, target = metric.preprocess(outputs, labels)
                            metric.update(preds, target)
                    if (
                        not self.training
                        and node_name in self.visualizers
                        and image is not None
                        and labels is not None
                    ):
                        for visualizer in self.visualizers[node_name]:  # type: ignore
                            viz = visualizer.forward(image, outputs, labels)
                            # TODO: smarter name
                            viz_name = visualizer.__class__.__name__
                            visualizations[node_name][viz_name] = viz

                # clean up results that are no longer needed
                if _cleanup and image is not None:
                    for key in list(computed.keys()):
                        if key not in self.outputs:
                            if all(pred in computed for pred in self.graph[key]):
                                del computed[key]

        outputs = {
            node_name: outputs
            for node_name, outputs in computed.items()
            if node_name in self.outputs
        }
        return LuxonisOutput(
            outputs=outputs, losses=losses, visualizations=visualizations
        )

    @contextmanager
    def export_mode(self):
        old_forward = self.forward

        def _export_forward(self, inputs: dict[NodeName, list[Tensor]]):
            return old_forward(inputs).outputs

        self.forward = partial(_export_forward, self)  # type: ignore
        yield
        self.forward = old_forward

    def export_onnx(self, save_path: str):
        inputs = {
            name: [torch.zeros(shape).cuda() for shape in shapes]
            for name, shapes in self.input_shapes.items()
        }
        with self.export_mode():
            torch.onnx.export(self, {"inputs": inputs}, save_path)  # type: ignore
        self._print_metric_warning("ONNX exported.")

    def training_step(self, train_batch: tuple, batch_idx: int):
        """Performs one step of training with provided batch"""
        inputs = train_batch[0]
        label_dict = train_batch[1]

        outputs = self.forward({"backbone": [inputs]}, label_dict)
        assert outputs.losses, "Losses are empty, check if you have defined any loss"
        # TODO: weighted sum here
        # TODO: manage sublosses
        loss = torch.zeros(1, device=self.device)
        for loss_name, (loss_value, sublosses) in outputs.losses.items():
            loss += loss_value
        self.training_step_outputs.append({"loss": loss.detach().cpu()})
        return loss

    def on_train_epoch_end(self):
        """Performs train epoch end operations"""
        epoch_train_loss = np.mean(
            [step_output["loss"] for step_output in self.training_step_outputs]
        )
        self.log("train_loss/loss", epoch_train_loss, sync_dist=True)
        self.training_step_outputs.clear()

    def validation_step(self, val_batch: tuple[Tensor, Labels], batch_idx: int):
        """Performs one step of validation with provided batch"""
        inputs = val_batch[0]
        unnormalize_img = self.cfg.get("train.preprocessing.normalize.active")
        normalize_params = self.cfg.get("train.preprocessing.normalize.params")
        image = preprocess_image(
            inputs,
            unnormalize_img=unnormalize_img,
            normalize_params=normalize_params,
        )[0]
        label_dict = val_batch[1]
        outputs = self.forward({"backbone": [inputs]}, label_dict, image=image)
        loss: Tensor = sum(map(lambda x: x[0], outputs.losses.values()))  # type: ignore
        for node_name, visualizations in outputs.visualizations.items():
            for viz_name, viz in visualizations.items():
                self.logger.log_image(
                    f"val_visualizations/{node_name}_{viz_name}_{batch_idx}",
                    viz.detach().cpu().numpy().transpose(1, 2, 0),
                    step=self.current_epoch,
                )

        step_output = {
            "loss": loss.detach().cpu(),
        }
        self.validation_step_outputs.append(step_output)

        return step_output

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

        metric_results = defaultdict(dict)  # used for printing to console
        self._print_metric_warning("Computing metrics on val subset ...")
        for node_name, metrics in self.metrics.items():
            for metric in metrics:
                metric_results[node_name][metric.name] = metric.compute()
                self.log(
                    f"val_metric/{node_name}_{metric.name}",
                    metric_results[node_name][metric.name],
                    sync_dist=True,
                )
                metric.reset()
        self._print_metric_warning("Metrics computed.")

        if self.cfg.get("trainer.verbose"):
            self._print_results(
                stage="Validation", loss=epoch_val_loss, metrics=metric_results
            )

        self.validation_step_outputs.clear()

    #   ---------------- From here down mostly unchanged ----------------

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
        """Checks if train eval should be performed on current epoch based on
        configured train_metrics_interval"""
        train_metrics_interval = self.cfg.get("train.train_metrics_interval")
        # add +1 to current_epoch because starting epoch is at 0
        return (
            train_metrics_interval != -1
            and (self.current_epoch + 1) % train_metrics_interval == 0
        )

    def _print_metric_warning(self, text: str):
        """Prints warning in the console for running metric computation
        (which can take quite long)"""
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
            print("Metrics:")
            pprint(metrics)
            print("----------")

    def configure_callbacks(self):
        """Configures Pytorch Lightning callbacks"""
        self.min_val_loss_checkpoints_path = f"{self.save_dir}/min_val_loss"
        self.best_val_metric_checkpoints_path = f"{self.save_dir}/best_val_metric"
        # self.main_metric = self.model.heads[0].main_metric

        loss_checkpoint = ModelCheckpoint(
            monitor="val_loss/loss",
            dirpath=self.min_val_loss_checkpoints_path,
            filename="loss={val_loss/loss:.4f}_" + "{epoch:02d}_" + self.model_name,
            auto_insert_metric_name=False,
            save_top_k=self.cfg.get("train.callbacks.model_checkpoint.save_top_k"),
            mode="min",
        )

        # metric_checkpoint = ModelCheckpoint(
        #     monitor=f"val_metric/{self.main_metric}",
        #     dirpath=self.best_val_metric_checkpoints_path,
        #     filename=self.main_metric
        #     + "={val_metric/"
        #     + self.main_metric
        #     + ":.4f}_loss={val_loss/loss:.4f}_{epoch:02d}_"
        #     + self.model_name,
        #     auto_insert_metric_name=False,
        #     save_top_k=self.cfg.get("train.callbacks.model_checkpoint.save_top_k"),
        #     mode="max",
        # )

        lr_monitor = LearningRateMonitor(logging_interval="step")
        module_freezer = ModuleFreezer(freeze_info=self.cfg.get("train.freeze_modules"))
        callbacks = [
            loss_checkpoint,
            # metric_checkpoint,
            lr_monitor,
            # annotation_checker,
            module_freezer,
        ]

        # used if we want to perform fine-grained debugging
        if self.cfg.get("train.callbacks.use_device_stats_monitor"):
            from pytorch_lightning.callbacks import DeviceStatsMonitor

            device_stats = DeviceStatsMonitor()
            callbacks.append(device_stats)

        if self.cfg.get("train.callbacks.early_stopping.active"):
            from pytorch_lightning.callbacks import EarlyStopping

            cfg_early_stopping = deepcopy(
                self.cfg.get("train.callbacks.early_stopping")
            )
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
                    CALLBACKS.get(custom_callback["name"])(
                        **custom_callback.get("params", {})
                    )
                )

        return callbacks

    def configure_optimizers(self):
        """Configures model optimizers and schedulers"""
        cfg_optimizer = self.cfg.get("train.optimizers")

        # config params + model parameters
        optim_params = {
            **cfg_optimizer["optimizer"].get("params", {}),
            "params": filter(lambda p: p.requires_grad, self.parameters()),
        }
        optimizer = OPTIMIZERS.get(cfg_optimizer["optimizer"]["name"])(**optim_params)

        # config params + optimizer
        scheduler_params = {
            **cfg_optimizer["scheduler"]["params"],
            "optimizer": optimizer,
        }
        scheduler = SCHEDULERS.get(cfg_optimizer["scheduler"]["name"])(
            **scheduler_params
        )

        return [optimizer], [scheduler]

    def load_checkpoint(self, path: str):
        """Loads checkpoint weights from provided path"""
        print(f"Loading weights from: {path}")
        fs = LuxonisFileSystem(path)
        checkpoint = torch.load(fs.read_to_byte_buffer())
        state_dict = checkpoint["state_dict"]
        self.load_state_dict(state_dict)
