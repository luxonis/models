import sys
import warnings
from typing import Annotated, Any, Literal

from luxonis_ml.data import BucketStorage, BucketType, LuxonisDataset, ValAugmentations
from luxonis_ml.utils import Config as LuxonisConfig
from pydantic import BaseModel, Field, field_serializer, model_validator
from torch.utils.data import DataLoader

from luxonis_train.utils.boxutils import anchors_from_dataset
from luxonis_train.utils.general import is_acyclic
from luxonis_train.utils.loaders import LuxonisLoaderTorch, collate_fn
from luxonis_train.utils.registry import MODELS


class AttachedModuleConfig(BaseModel):
    name: str
    attached_to: str
    override_name: str | None = None
    params: dict[str, Any] = {}


class LossModuleConfig(AttachedModuleConfig):
    weight: float = 1.0


class MetricModuleConfig(AttachedModuleConfig):
    is_main_metric: bool = False


class ModelNodeConfig(BaseModel):
    name: str
    override_name: str | None = None
    inputs: list[str] = []
    params: dict[str, Any] = {}
    frozen: bool = False


class ModelConfig(BaseModel):
    name: str
    predefined_model: str | None = None
    weights: str | None = None
    params: dict[str, Any] = {}
    nodes: list[ModelNodeConfig] = []
    losses: list[LossModuleConfig] = []
    metrics: list[MetricModuleConfig] = []
    visualizers: list[AttachedModuleConfig] = []
    outputs: list[str] = []

    @model_validator(mode="after")
    def check_predefined_model(self) -> "ModelConfig":
        if self.predefined_model:
            # TODO:  Unified LuxonisLogger instead of warnings
            warnings.warn("Loading predefined model type")
            self.nodes += MODELS.get(self.predefined_model)(**self.params)

        elif self.params:
            warnings.warn(
                "Model-wise parameters ignored as no `predefined_model` specified."
            )

        return self

    @model_validator(mode="after")
    def check_graph(self) -> "ModelConfig":
        graph = {node.override_name or node.name: node.inputs for node in self.nodes}
        if not is_acyclic(graph):
            raise ValueError("Model graph is not acyclic.")
        if not self.outputs:
            outputs = []  # nodes which are not inputs to any nodes
            inputs = set(node_name for node in self.nodes for node_name in node.inputs)
            for node in self.nodes:
                name = node.override_name or node.name
                if name not in inputs:
                    outputs.append(name)
            self.outputs = outputs
        if not self.outputs:
            raise ValueError("No outputs specified.")
        return self

    model_config = {
        "json_schema_extra": {
            "if": {"properties": {"predefined_model": {"type": "null"}}},
            "then": {
                "allOf": [
                    {"properties": {"nodes": {"type": "array"}}},
                ],
            },
        }
    }


class TrainerConfig(BaseModel):
    accelerator: Literal["auto", "cpu", "gpu"] = "auto"
    devices: int | list[int] | str = "auto"
    strategy: Literal["auto", "ddp"] = "auto"
    num_sanity_val_steps: int = 2
    profiler: Literal["simple", "advanced"] | None = None
    verbose: bool = True


class LoggerConfig(BaseModel):
    project_name: str | None = None
    project_id: str | None = None
    run_name: str | None = None
    run_id: str | None = None
    save_directory: str = "output"
    is_tensorboard: bool = True
    is_wandb: bool = False
    wandb_entity: str | None = None
    is_mlflow: bool = False
    logged_hyperparams: list[str] = ["train.epochs", "train.batch_size"]

    # Can also add extra validation based on Tracker class


class DatasetConfig(BaseModel):
    dataset_name: str | None = None
    dataset_id: str | None = None
    team_name: str | None = None
    team_id: str | None = None
    bucket_type: BucketType = BucketType.INTERNAL
    bucket_storage: BucketStorage = BucketStorage.LOCAL
    json_mode: bool = False
    train_view: str = "train"
    val_view: str = "val"
    test_view: str = "test"

    @model_validator(mode="after")
    def check_dataset_params(self) -> "DatasetConfig":
        if not self.dataset_name and not self.dataset_id:
            raise ValueError("Must provide either `dataset_name` or `dataset_id`.")
        return self

    @field_serializer("bucket_storage", "bucket_type")
    def get_eunm_value(self, v, _) -> str:
        return str(v.value)

    model_config = {
        "json_schema_extra": {
            "anyOf": [
                {
                    "allOf": [
                        {"required": ["dataset_name"]},
                        {"properties": {"dataset_name": {"type": "string"}}},
                    ]
                },
                {
                    "allOf": [
                        {"required": ["dataset_id"]},
                        {"properties": {"dataset_id": {"type": "string"}}},
                    ]
                },
            ]
        },
    }


class NormalizeAugmentationConfig(BaseModel):
    active: bool = True
    params: dict[str, Any] = {}


class AugmentationConfig(BaseModel):
    name: str
    params: dict[str, Any] = {}


class PreprocessingConfig(BaseModel):
    train_image_size: Annotated[
        list[int], Field(default=[256, 256], min_length=2, max_length=2)
    ] = [256, 256]
    keep_aspect_ratio: bool = True
    train_rgb: bool = True
    normalize: NormalizeAugmentationConfig = NormalizeAugmentationConfig()
    augmentations: list[AugmentationConfig] = []

    @model_validator(mode="after")
    def check_normalize(self) -> "PreprocessingConfig":
        if self.normalize.active:
            self.augmentations.append(
                AugmentationConfig(name="Normalize", params=self.normalize.params)
            )
        return self


class CallbackConfig(BaseModel):
    name: str
    active: bool = True
    params: dict[str, Any] = {}


class OptimizerConfig(BaseModel):
    name: str = "Adam"
    params: dict[str, Any] = {}


class SchedulerConfig(BaseModel):
    name: str = "ConstantLR"
    params: dict[str, Any] = {}


class TrainConfig(BaseModel):
    preprocessing: PreprocessingConfig = PreprocessingConfig()

    batch_size: int = 32
    accumulate_grad_batches: int = 1
    use_weighted_sampler: bool = False
    epochs: int = 100
    num_workers: int = 2
    train_metrics_interval: int = -1
    validation_interval: int = 1
    num_log_images: int = 4
    skip_last_batch: bool = True
    main_head_index: int = 0
    use_rich_text: bool = True
    log_sub_losses: bool = True
    save_top_k: int = 3

    callbacks: list[CallbackConfig] = []

    optimizer: OptimizerConfig = OptimizerConfig()
    scheduler: SchedulerConfig = SchedulerConfig()

    @model_validator(mode="after")
    def check_num_workes_platform(self) -> "TrainConfig":
        if (
            sys.platform == "win32" or sys.platform == "darwin"
        ) and self.num_workes != 0:
            self.num_workers = 0
            warnings.warn(
                "Setting `num_workers` to 0 because of platform compatibility."
            )
        return self


class OnnxExportConfig(BaseModel):
    opset_version: int = 12
    dynamic_axes: dict[str, Any] | None = None


class BlobconverterExportConfig(BaseModel):
    active: bool = False
    shaves: int = 6


class ExportConfig(BaseModel):
    export_save_directory: str = "output_export"
    input_shape: list[int] | None = None
    export_model_name: str = "model"
    data_type: Literal["INT8", "FP16", "FP32"] = "FP16"
    reverse_input_channels: bool = True
    scale_values: list[float] | None = None
    mean_values: list[float] | None = None
    onnx: OnnxExportConfig = OnnxExportConfig()
    blobconverter: BlobconverterExportConfig = BlobconverterExportConfig()
    upload_directory: str | None = None

    @model_validator(mode="after")
    def check_values(self) -> "ExportConfig":
        def pad_values(values: float | list[float] | None):
            if values is None:
                return None
            if isinstance(values, float):
                return [values] * 3

        self.scale_values = pad_values(self.scale_values)
        self.mean_values = pad_values(self.mean_values)
        return self


class StorageConfig(BaseModel):
    active: bool = True
    storage_type: Literal["local", "remote"] = "local"


class TunerConfig(BaseModel):
    study_name: str = "test-study"
    use_pruner: bool = True
    n_trials: int = 3
    timeout: int | None = None
    storage: StorageConfig = StorageConfig()
    params: Annotated[
        dict[str, list[str | int | float | bool]], Field(default={}, min_length=1)
    ] = {}

    model_config = {"json_schema_extra": {"required": ["params"]}}


class Config(LuxonisConfig):
    model: ModelConfig
    dataset: DatasetConfig
    trainer: TrainerConfig = TrainerConfig()
    logger: LoggerConfig = LoggerConfig()
    train: TrainConfig = TrainConfig()
    exporter: ExportConfig = ExportConfig()
    tuner: TunerConfig = TunerConfig()

    @model_validator(mode="before")
    @classmethod
    def check_tuner_init(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if data.get("tuner") and not data.get("tuner", {}).get("params"):
                del data["tuner"]
                warnings.warn(
                    "`tuner` block specified but no `tuner.params`. If tyring to tune values you have to specify at least one parameter"
                )
        return data

    def _validate(self) -> None:
        """Performs any additional validation on the top level after the fact."""
        for node in self.model.nodes:
            if (
                node.name == "ImplicitKeypointBBoxHead"
                and node.params.get("anchors") is None
            ):
                warnings.warn("Generating anchors for ImplicitKeypointBBoxHead")
                node.params["anchors"] = self._autogenerate_anchors(node)

        if self._fs is not None and self._fs.is_mlflow:
            warnings.warn("Setting `project_id` and `run_id` to config's MLFlow run")
            self.logger.project_id = self._fs.experiment_id
            self.logger.run_id = self._fs.run_id

    def _autogenerate_anchors(self, head: ModelNodeConfig) -> list[list[float]]:
        """Automatically generates anchors for the provided dataset.

        Args:
            head (ModelNodeConfig): Config of the head where anchors will be used

        Returns:
            list[list[float]]: list of anchors in [-1,6] format
        """

        dataset = LuxonisDataset(
            dataset_name=self.dataset.dataset_name,
            team_id=self.dataset.team_id,
            dataset_id=self.dataset.dataset_id,
            bucket_type=self.dataset.bucket_type,
            bucket_storage=self.dataset.bucket_storage,
        )
        val_augmentations = ValAugmentations(
            image_size=self.train.preprocessing.train_image_size,
            augmentations=[{"name": "Normalize", "params": {}}],
            train_rgb=self.train.preprocessing.train_rgb,
            keep_aspect_ratio=self.train.preprocessing.keep_aspect_ratio,
        )
        loader = LuxonisLoaderTorch(
            dataset,
            view=self.dataset.train_view,
            augmentations=val_augmentations,
        )
        pytorch_loader = DataLoader(
            loader,
            batch_size=self.train.batch_size,
            num_workers=self.train.num_workers,
            collate_fn=collate_fn,
        )
        num_heads = head.params.get("num_heads", 3)
        proposed_anchors = anchors_from_dataset(pytorch_loader, n_anchors=num_heads * 3)
        return proposed_anchors.reshape(-1, 6).tolist()
