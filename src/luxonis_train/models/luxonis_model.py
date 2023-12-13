import logging
from collections import defaultdict
from collections.abc import Mapping
from typing import Literal, cast

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    RichModelSummary,
)
from pytorch_lightning.utilities import rank_zero_only  # type: ignore
from torch import Size, Tensor, nn

from luxonis_train.attached_modules import (
    BaseAttachedModule,
    BaseLoss,
    BaseMetric,
    BaseVisualizer,
)
from luxonis_train.attached_modules.visualizers import (
    combine_visualizations,
    get_unnormalized_images,
)
from luxonis_train.callbacks import (
    LuxonisProgressBar,
    ModuleFreezer,
)
from luxonis_train.nodes import BaseNode
from luxonis_train.utils.config import AttachedModuleConfig, Config
from luxonis_train.utils.general import (
    DatasetMetadata,
    get_shape_packet,
    traverse_graph,
)
from luxonis_train.utils.registry import CALLBACKS, OPTIMIZERS, SCHEDULERS, Registry
from luxonis_train.utils.tracker import LuxonisTrackerPL
from luxonis_train.utils.types import Kwargs, Labels, Packet

from .luxonis_output import LuxonisOutput


class LuxonisModel(pl.LightningModule):
    """Class representing the entire model.

    This class keeps track of the model graph, nodes, and attached modules.
    The model topology is defined as an acyclic graph of nodes.
    The graph is saved as a dictionary of predecessors.

    Attributes:
        save_dir (str): Directory to save checkpoints and logs.

        nodes (nn.ModuleDict[str, LuxonisModule]):
          Nodes of the model. Keys are node names, unique for each node.
          Values are instances of `LuxonisModule`.

        graph (dict[str, list[str]]): Graph of the model in a format of a dictionary
          of predecessors. Keys are node names, values are inputs to the
          node (list of node names). Nodes with no inputs are considered
          inputs of the whole model.

        loss_weights (dict[str, float]): Dictionary of loss weights.
          Keys are loss names, values are weights.

        input_shapes (dict[str, list[Shape]]): Dictionary of input shapes.
          Keys are node names, values are lists of shapes
          (understood as shapes of the "feature" field in `Packet[Tensor]`).

        outputs (list[str]): List of output node names.

        losses (nn.ModuleDict[str, nn.ModuleDict[str, LuxonisLoss]]): Nested dictionary
          of losses used in the model. Each node can have multiple losses attached.
          The first key identifies the node, the second key identifies the specific loss.

        visualizers (dict[str, dict[str, LuxonisVisualizer]]): Dictionary
          of visualizers to be used with the model. For each node name,
          there is a dictionary where keys are visualizer names (unique) and values
          are `LuxonisVisualizer` instances. Defaults to empty dictionary.

        metrics (dict[str, dict[str, LuxonisMetric]]): Dictionary of
          metrics to be used with the model. For each node name, there is
          a dictionary where keys are metric names (unique) and values are
          `LuxonisMetric` instances.

        main_metric (str, optional): Name of the main metric to be used for
            model checkpointing. If not set, the model with the best metric score
            won't be saved.
    """

    _trainer: pl.Trainer
    logger: LuxonisTrackerPL

    def __init__(
        self,
        cfg: Config,
        save_dir: str,
        input_shape: list[int] | Size,
        dataset_metadata: DatasetMetadata | None = None,
        **kwargs,
    ):
        """Constructs an instance of `LuxonisModel` from `Config`.

        Args:
            cfg (Config): Config object.
            save_dir (str): Directory to save checkpoints.
            input_shape (list[int] | Size): Shape of the input tensor.
            dataset_metadata (DatasetMetadata, optional): Dataset metadata.
            **kwargs: Additional arguments to pass to the parent class.
        """
        super().__init__(**kwargs)

        self._export: bool = False

        self.cfg = cfg
        self.original_in_shape = Size(input_shape)
        self.dataset_metadata = dataset_metadata or DatasetMetadata()
        self.frozen_nodes: list[nn.Module] = []
        self.graph: dict[str, list[str]] = {}
        self.input_shapes: dict[str, list[Size]] = {}
        self.loss_weights: dict[str, float] = {}
        self.main_metric: str | None = None
        self.save_dir = save_dir
        self.test_step_outputs: list[Mapping[str, Tensor | float | int]] = []
        self.training_step_outputs: list[Mapping[str, Tensor | float | int]] = []
        self.validation_step_outputs: list[Mapping[str, Tensor | float | int]] = []
        self.losses: dict[str, dict[str, BaseLoss]] = defaultdict(dict)
        self.metrics: dict[str, dict[str, BaseMetric]] = defaultdict(dict)
        self.visualizers: dict[str, dict[str, BaseVisualizer]] = defaultdict(dict)
        self.logging_logger = logging.getLogger(__name__)

        self._logged_images = 0

        frozen_nodes: list[str] = []
        nodes: dict[str, tuple[type[BaseNode], Kwargs]] = {}

        for node_cfg in self.cfg.model.nodes:
            node_name = node_cfg.name
            if node_cfg.frozen:
                frozen_nodes.append(node_name)
            Node = BaseNode.REGISTRY.get(node_name)
            node_name = node_cfg.override_name or node_name
            nodes[node_name] = (Node, node_cfg.params)
            if not node_cfg.inputs:
                self.input_shapes[node_name] = [Size(input_shape)]
            self.graph[node_name] = node_cfg.inputs

        self.nodes = self._initiate_nodes(nodes)

        for loss_cfg in self.cfg.model.losses:
            loss_name, _ = self._init_attached_module(
                loss_cfg, BaseLoss.REGISTRY, self.losses
            )
            self.loss_weights[loss_name] = loss_cfg.weight

        for metric_cfg in self.cfg.model.metrics:
            metric_name, node_name = self._init_attached_module(
                metric_cfg, BaseMetric.REGISTRY, self.metrics
            )
            if metric_cfg.is_main_metric:
                if self.main_metric is not None:
                    raise ValueError(
                        "Multiple main metrics defined. Only one is allowed."
                    )
                self.main_metric = f"{node_name}/{metric_name}"

        for visualizer_cfg in self.cfg.model.visualizers:
            self._init_attached_module(
                visualizer_cfg, BaseVisualizer.REGISTRY, self.visualizers
            )

        self.outputs = self.cfg.model.outputs
        self.frozen_nodes = [self.nodes[name] for name in frozen_nodes]
        self.losses = self._to_module_dict(self.losses)  # type: ignore
        self.metrics = self._to_module_dict(self.metrics)  # type: ignore
        self.visualizers = self._to_module_dict(self.visualizers)  # type: ignore

        self.load_checkpoint(self.cfg.model.weights)

    def _initiate_nodes(
        self,
        nodes: dict[str, tuple[type[BaseNode], Kwargs]],
    ) -> nn.ModuleDict:
        """Initializes all the nodes in the model.

        Traverses the graph and initiates each node using outputs of
        the preceding nodes.


        Args:
            nodes (dict[str, tuple[type[LuxonisNode], Kwargs]]): Dictionary of nodes
              to be initiated. Keys are node names, values are tuples of node class
              and node kwargs.

        Returns:
            nn.ModuleDict[str, LuxonisNode]: Dictionary of initiated nodes.
        """
        initiated_nodes: dict[str, BaseNode] = {}

        dummy_outputs: dict[str, Packet[Tensor]] = {
            f"__{node_name}_input__": {
                "features": [torch.zeros(2, *shape[1:]) for shape in shapes]
            }
            for node_name, shapes in self.input_shapes.items()
        }

        for node_name, (Node, node_kwargs), node_input_names, _ in traverse_graph(
            self.graph, nodes
        ):
            node_input_shapes: list[Packet[Size]] = []
            node_dummy_inputs: list[Packet[Tensor]] = []

            if not node_input_names:
                node_input_names = [f"__{node_name}_input__"]

            for node_input_name in node_input_names:
                dummy_output = dummy_outputs[node_input_name]
                shape_packet = get_shape_packet(dummy_output)
                node_input_shapes.append(shape_packet)
                node_dummy_inputs.append(dummy_output)

                node = Node(
                    input_shapes=node_input_shapes,
                    original_in_shape=self.original_in_shape,
                    dataset_metadata=self.dataset_metadata,
                    **node_kwargs,
                )
                node_outputs = node.run(node_dummy_inputs)

                dummy_outputs[node_name] = node_outputs
                initiated_nodes[node_name] = node

        return nn.ModuleDict(initiated_nodes)

    def forward(
        self,
        inputs: Tensor,
        labels: Labels | None = None,
        images: Tensor | None = None,
        *,
        compute_loss: bool = True,
        compute_metrics: bool = False,
        compute_visualizations: bool = False,
    ) -> LuxonisOutput:
        """Forward pass of the model.

        Traverses the graph and step-by-step computes the outputs of each node.
        Each next node is computed only when all of its predecessors are computed.
        Once the outputs are not needed anymore, they are removed from the memory.

        Args:
            inputs (Tensor): Input tensor.
            labels (Labels, optional): Labels dictionary. Defaults to None.
            image (Tensor, optional): Canvas tensor for visualizers. Defaults to None.

            compute_loss (bool, optional): Whether to compute losses. Defaults to True.
            compute_metrics (bool, optional): Whether to update metrics.
              Defaults to True.
            compute_visualizations (bool, optional): Whether to compute
              visualizations. Defaults to False.

        Returns:
            LuxonisOutput: Output of the model.
        """
        input_node_name = list(self.input_shapes.keys())[0]
        input_dict = {input_node_name: [inputs]}

        losses: dict[
            str, dict[str, Tensor | tuple[Tensor, dict[str, Tensor]]]
        ] = defaultdict(dict)
        visualizations: dict[str, dict[str, Tensor]] = defaultdict(dict)

        computed: dict[str, Packet[Tensor]] = {
            f"__{node_name}_input__": {"features": input_tensors}
            for node_name, input_tensors in input_dict.items()
        }
        for node_name, node, input_names, unprocessed in traverse_graph(
            self.graph, cast(dict[str, BaseNode], self.nodes)
        ):
            # Special input for the first node. Will be changed when
            # multiple inputs will be supported in `luxonis-ml.data`.
            if not input_names:
                input_names = [f"__{node_name}_input__"]

            node_inputs = [computed[pred] for pred in input_names]
            outputs = node.run(node_inputs)
            computed[node_name] = outputs

            if compute_loss and node_name in self.losses and labels is not None:
                for loss_name, loss in self.losses[node_name].items():
                    losses[node_name][loss_name] = loss.run(outputs, labels)

            if compute_metrics and node_name in self.metrics and labels is not None:
                for metric in self.metrics[node_name].values():
                    metric.run_update(outputs, labels)

            if (
                compute_visualizations
                and node_name in self.visualizers
                and images is not None
                and labels is not None
            ):
                for viz_name, visualizer in self.visualizers[node_name].items():
                    viz = combine_visualizations(
                        visualizer.run(
                            images,
                            images,
                            outputs,
                            labels,
                        ),
                    )
                    visualizations[node_name][viz_name] = viz

            for computed_name in list(computed.keys()):
                if computed_name in self.outputs:
                    continue
                for node_name in unprocessed:
                    if computed_name in self.graph[node_name]:
                        break
                else:
                    del computed[computed_name]

        outputs_dict = {
            node_name: outputs
            for node_name, outputs in computed.items()
            if node_name in self.outputs
        }

        return LuxonisOutput(
            outputs=outputs_dict, losses=losses, visualizations=visualizations
        )

    def compute_metrics(self) -> dict[str, dict[str, Tensor]]:
        """Computes metrics and returns their values.

        Goes through all metrics in the `metrics` attribute and computes their values.
        After the computation, the metrics are reset.

        Returns:
            dict[str, dict[str, Tensor]]: Dictionary of computed metrics.
              Each node can have multiple metrics attached. The first key identifies
              the node, the second key identifies the specific metric.
        """
        metric_results: dict[str, dict[str, Tensor]] = defaultdict(dict)
        for node_name, metrics in self.metrics.items():
            for metric_name, metric in metrics.items():
                match metric.compute():
                    case (Tensor(data=metric_value), dict(submetrics)):
                        computed_submetrics = {
                            metric_name: metric_value,
                        } | submetrics
                    case Tensor(data=metric_value):
                        computed_submetrics = {metric_name: metric_value}
                    case dict(submetrics):
                        computed_submetrics = submetrics
                    case unknown:
                        raise ValueError(
                            f"Metric {metric_name} returned unexpected value of "
                            f"type {type(unknown)}."
                        )
                metric.reset()
                metric_results[node_name] |= computed_submetrics
        return metric_results

    def export_onnx(self, save_path: str, **kwargs) -> list[str]:
        """Exports the model to ONNX format.

        Args:
            save_path (str): Path where the exported model will be saved.
            **kwargs: Additional arguments for the `torch.onnx.export` method.

        Returns:
            list[str]: List of output names.
        """

        inputs = {
            name: [torch.zeros(shape).to(self.device) for shape in shapes]
            for name, shapes in self.input_shapes.items()
        }

        # TODO: multiple inputs
        inp = list(inputs.values())[0][0]

        for module in self.modules():
            if isinstance(module, BaseNode):
                module.set_export_mode()

        outputs = self.forward(inp.clone()).outputs
        output_order = sorted(
            [
                (node_name, output_name, i)
                for node_name, outs in outputs.items()
                for output_name, out in outs.items()
                for i in range(len(out))
            ]
        )
        output_names = [
            f"{node_name}/{output_name}/{i}"
            for node_name, output_name, i in output_order
        ]

        old_forward = self.forward

        def export_forward(inputs) -> tuple[Tensor, ...]:
            outputs = old_forward(
                inputs,
                None,
                compute_loss=False,
                compute_metrics=False,
                compute_visualizations=False,
            ).outputs
            return tuple(
                outputs[node_name][output_name][i]
                for node_name, output_name, i in output_order
            )

        self.forward = export_forward  # type: ignore
        if "output_names" not in kwargs:
            kwargs["output_names"] = output_names

        self.to_onnx(save_path, inp, **kwargs)

        self.forward = old_forward  # type: ignore

        for module in self.modules():
            if isinstance(module, BaseNode):
                module.set_export_mode(False)

        self.logging_logger.info(f"Model exported to {save_path}")
        return output_names

    def process_losses(
        self,
        losses_dict: dict[str, dict[str, Tensor | tuple[Tensor, dict[str, Tensor]]]],
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Processes individual losses from the model run.

        Goes over the computed losses and computes the final loss as a weighted sum
        of all the losses.

        Args:
            losses_dict (dict[str, dict[str, Tensor | tuple[Tensor, dict[str, Tensor]]]]):
              Dictionary of computed losses. Each node can have multiple losses attached.
              The first key identifies the node, the second key identifies the specific loss.
              Values are either single tensors or tuples of tensors and sublosses.

        Returns:
            tuple[Tensor, dict[str, Tensor]]: Tuple of final loss and dictionary of
              processed sublosses. The dictionary is in a format of
              {loss_name: loss_value}.
        """
        final_loss = torch.zeros(1, device=self.device)
        training_step_output: dict[str, Tensor] = {}
        for node_name, losses in losses_dict.items():
            for loss_name, loss_values in losses.items():
                if isinstance(loss_values, tuple):
                    loss, sublosses = loss_values
                else:
                    loss = loss_values
                    sublosses = {}

                loss *= self.loss_weights[loss_name]
                final_loss += loss
                training_step_output[
                    f"loss/{node_name}/{loss_name}"
                ] = loss.detach().cpu()
                if self.cfg.trainer.log_sub_losses and sublosses:
                    for subloss_name, subloss_value in sublosses.items():
                        training_step_output[
                            f"loss/{node_name}/{loss_name}/{subloss_name}"
                        ] = subloss_value.detach().cpu()
        training_step_output["loss"] = final_loss.detach().cpu()
        return final_loss, training_step_output

    def training_step(self, train_batch: tuple[Tensor, Labels]) -> Tensor:
        """Performs one step of training with provided batch."""
        outputs = self.forward(*train_batch)
        assert outputs.losses, "Losses are empty, check if you have defined any loss"

        loss, training_step_output = self.process_losses(outputs.losses)
        self.training_step_outputs.append(training_step_output)
        return loss

    def validation_step(self, val_batch: tuple[Tensor, Labels]) -> dict[str, Tensor]:
        """Performs one step of validation with provided batch."""
        return self._evaluation_step("val", val_batch)

    def test_step(self, test_batch: tuple[Tensor, Labels]) -> dict[str, Tensor]:
        """Performs one step of testing with provided batch."""
        return self._evaluation_step("test", test_batch)

    def on_train_epoch_end(self) -> None:
        """Performs train epoch end operations."""
        epoch_train_losses = self._average_losses(self.training_step_outputs)
        for module in self.modules():
            if isinstance(module, (BaseNode, BaseLoss)):
                module._epoch = self.current_epoch

        for key, value in epoch_train_losses.items():
            self.log(f"train/{key}", value, sync_dist=True)

        self.training_step_outputs.clear()

    def on_validation_epoch_end(self) -> None:
        """Performs validation epoch end operations."""
        return self._evaluation_epoch_end("val")

    def on_test_epoch_end(self) -> None:
        """Performs test epoch end operations."""
        return self._evaluation_epoch_end("test")

    def get_status(self) -> tuple[int, int]:
        """Returns current epoch and number of all epochs."""
        return self.current_epoch, self.cfg.trainer.epochs

    def get_status_percentage(self) -> float:
        """Returns percentage of current training, takes into account early stopping."""
        if self._trainer.early_stopping_callback:
            # model haven't yet stop from early stopping callback
            if self._trainer.early_stopping_callback.stopped_epoch == 0:
                return (self.current_epoch / self.cfg.trainer.epochs) * 100
            else:
                return 100.0
        else:
            return (self.current_epoch / self.cfg.trainer.epochs) * 100

    def _evaluation_step(
        self, mode: Literal["test", "val"], batch: tuple[Tensor, Labels]
    ) -> dict[str, Tensor]:
        inputs, labels = batch
        images = None
        if self._logged_images < self.cfg.trainer.num_log_images:
            images = get_unnormalized_images(self.cfg, inputs)
        outputs = self.forward(
            inputs,
            labels,
            images=images,
            compute_metrics=True,
            compute_visualizations=True,
        )

        _, step_output = self.process_losses(outputs.losses)
        self.validation_step_outputs.append(step_output)

        logged_images = self._logged_images
        for node_name, visualizations in outputs.visualizations.items():
            for viz_name, viz_batch in visualizations.items():
                logged_images = self._logged_images
                for viz in viz_batch:
                    if logged_images >= self.cfg.trainer.num_log_images:
                        break
                    self.logger.log_image(
                        f"{mode}/visualizations/{node_name}/{viz_name}/{logged_images}",
                        viz.detach().cpu().numpy().transpose(1, 2, 0),
                        step=self.current_epoch,
                    )
                    logged_images += 1
        self._logged_images = logged_images

        return step_output

    def _evaluation_epoch_end(self, mode: Literal["test", "val"]) -> None:
        epoch_val_losses = self._average_losses(self.validation_step_outputs)

        for key, value in epoch_val_losses.items():
            self.log(f"{mode}/{key}", value, sync_dist=True)

        metric_results: dict[str, dict[str, float]] = defaultdict(dict)
        self.logging_logger.info(f"Computing metrics on {mode} subset ...")
        computed_metrics = self.compute_metrics()
        self.logging_logger.info("Metrics computed.")
        for node_name, metrics in computed_metrics.items():
            for metric_name, metric_value in metrics.items():
                metric_results[node_name][metric_name] = metric_value.cpu().item()
                self.log(
                    f"{mode}/metric/{node_name}/{metric_name}",
                    metric_value,
                    sync_dist=True,
                )

        if self.cfg.trainer.verbose:
            self._print_results(
                stage="Validation" if mode == "val" else "Test",
                loss=epoch_val_losses["loss"],
                metrics=metric_results,
            )

        self.validation_step_outputs.clear()
        self._logged_images = 0

    def configure_callbacks(self) -> list[pl.Callback]:
        """Configures Pytorch Lightning callbacks."""
        self.min_val_loss_checkpoints_path = f"{self.save_dir}/min_val_loss"
        self.best_val_metric_checkpoints_path = f"{self.save_dir}/best_val_metric"
        model_name = self.cfg.model.name

        callbacks: list[pl.Callback] = []

        callbacks.append(
            ModelCheckpoint(
                monitor="val/loss",
                dirpath=self.min_val_loss_checkpoints_path,
                filename=f"{model_name}_loss={{val/loss:.4f}}_{{epoch:02d}}",
                auto_insert_metric_name=False,
                save_top_k=self.cfg.trainer.save_top_k,
                mode="min",
            )
        )

        if self.main_metric is not None:
            main_metric = self.main_metric.replace("/", "_")
            callbacks.append(
                ModelCheckpoint(
                    monitor=f"val/metric/{self.main_metric}",
                    dirpath=self.best_val_metric_checkpoints_path,
                    filename=f"{model_name}_{main_metric}={{val/metric/{self.main_metric}:.4f}}"
                    f"_loss={{val/loss:.4f}}_{{epoch:02d}}",
                    auto_insert_metric_name=False,
                    save_top_k=self.cfg.trainer.save_top_k,
                    mode="max",
                )
            )

        if self.frozen_nodes:
            callbacks.append(ModuleFreezer(self.frozen_nodes))

        if self.cfg.use_rich_text:
            callbacks.append(RichModelSummary(max_depth=2))

        for callback in self.cfg.trainer.callbacks:
            if callback.active:
                callbacks.append(CALLBACKS.get(callback.name)(**callback.params))

        return callbacks

    def configure_optimizers(
        self,
    ) -> tuple[list[torch.optim.Optimizer], list[nn.Module]]:
        """Configures model optimizers and schedulers."""
        cfg_optimizer = self.cfg.trainer.optimizer
        cfg_scheduler = self.cfg.trainer.scheduler

        optim_params = cfg_optimizer.params | {
            "params": filter(lambda p: p.requires_grad, self.parameters()),
        }
        optimizer = OPTIMIZERS.get(cfg_optimizer.name)(**optim_params)

        scheduler_params = cfg_scheduler.params | {"optimizer": optimizer}
        scheduler = SCHEDULERS.get(cfg_scheduler.name)(**scheduler_params)

        return [optimizer], [scheduler]

    def load_checkpoint(self, path: str | None) -> None:
        """Loads checkpoint weights from provided path.

        Loads the checkpoints gracefully, ignoring keys that are not found
        in the model state dict or in the checkpoint.

        Args:
            path (str | None): Path to the checkpoint. If None, no checkpoint
              will be loaded.
        """
        if path is None:
            return
        checkpoint = torch.load(path, map_location=self.device)
        if "state_dict" not in checkpoint:
            raise ValueError("Checkpoint does not contain state_dict.")
        state_dict = {}
        self_state_dict = self.state_dict()
        for key, value in checkpoint["state_dict"].items():
            if key not in self_state_dict.keys():
                self.logging_logger.warning(
                    f"Key `{key}` from checkpoint not found in model state dict."
                )
            else:
                state_dict[key] = value

        for key in self_state_dict:
            if key not in state_dict.keys():
                self.logging_logger.warning(f"Key `{key}` was not found in checkpoint.")
            else:
                try:
                    self_state_dict[key].copy_(state_dict[key])
                except Exception:
                    self.logging_logger.warning(
                        f"Key `{key}` from checkpoint could not be loaded into model."
                    )

        self.logging_logger.info(f"Loaded checkpoint from {path}.")

    def _init_attached_module(
        self,
        cfg: AttachedModuleConfig,
        registry: Registry,
        storage: Mapping[str, Mapping[str, BaseAttachedModule]],
    ) -> tuple[str, str]:
        Module = registry.get(cfg.name)
        module_name = cfg.override_name or cfg.name
        node_name = cfg.attached_to
        module = Module(**cfg.params, node=self.nodes[node_name])
        storage[node_name][module_name] = module  # type: ignore
        return module_name, node_name

    @staticmethod
    def _to_module_dict(modules: dict[str, dict[str, nn.Module]]) -> nn.ModuleDict:
        return nn.ModuleDict(
            {
                node_name: nn.ModuleDict(node_modules)
                for node_name, node_modules in modules.items()
            }
        )

    @property
    def _progress_bar(self) -> LuxonisProgressBar:
        return cast(LuxonisProgressBar, self._trainer.progress_bar_callback)

    @rank_zero_only
    def _print_results(
        self, stage: str, loss: float, metrics: dict[str, dict[str, float]]
    ) -> None:
        """Prints validation metrics in the console."""

        self.logging_logger.info(f"{stage} loss: {loss:.4f}")

        if self.cfg.use_rich_text:
            self._progress_bar.print_results(stage=stage, loss=loss, metrics=metrics)
        else:
            for node_name, node_metrics in metrics.items():
                for metric_name, metric_value in node_metrics.items():
                    self.logging_logger.info(
                        f"{stage} metric: {node_name}/{metric_name}: {metric_value:.4f}"
                    )

        if self.main_metric is not None:
            main_metric_node, main_metric_name = self.main_metric.split("/")
            main_metric = metrics[main_metric_node][main_metric_name]
            self.logging_logger.info(
                f"{stage} main metric ({self.main_metric}): {main_metric:.4f}"
            )

    def _is_train_eval_epoch(self) -> bool:
        """Checks if train eval should be performed on current epoch based on configured
        train_metrics_interval."""
        train_metrics_interval = self.cfg.trainer.train_metrics_interval
        # add +1 to current_epoch because starting epoch is at 0
        return (
            train_metrics_interval != -1
            and (self.current_epoch + 1) % train_metrics_interval == 0
        )

    def _average_losses(
        self, step_outputs: list[Mapping[str, Tensor | float | int]]
    ) -> dict[str, float]:
        avg_losses: dict[str, float] = defaultdict(float)

        for step_output in step_outputs:
            for key, value in step_output.items():
                avg_losses[key] += float(value)

        for key in avg_losses:
            avg_losses[key] /= len(step_outputs)
        return avg_losses
