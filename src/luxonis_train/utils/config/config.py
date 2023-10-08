import warnings
from pydantic import BaseModel, Field, model_validator
from typing import Optional, Dict, Any, List, Tuple, Literal, Union, Annotated


# ---- MODEL CONFIG ----
class ModelModuleConfig(BaseModel):
    name: str
    params: Dict[str, Any] = {}


class LossModuleConfig(BaseModel):
    name: str
    params: Dict[str, Any] = {}


class ModelHeadConfig(BaseModel):
    name: str
    params: Dict[str, Any] = {}
    loss: LossModuleConfig


class ModelConfig(BaseModel):
    name: str
    predefined_model: Optional[str] = None
    pretrained: Optional[str] = None
    params: Dict[str, Any] = {}
    backbone: Optional[ModelModuleConfig] = None
    neck: Optional[ModelModuleConfig] = None
    heads: Optional[Annotated[List[ModelHeadConfig], Field(min_length=1)]] = None
    additional_heads: Optional[List[ModelHeadConfig]] = None

    @model_validator(mode="after")
    def check_predefined_model(self) -> "ModelConfig":
        if self.predefined_model:
            if self.backbone or self.neck or self.heads:
                warnings.warn(
                    "Specified backbone, neck or head ignored as `predefined_model` specified."
                )

            warnings.warn("Loading predefined model type")
            backbone, neck, heads = self._generate_predefined_model()
            self.backbone = backbone
            self.neck = neck
            self.heads = heads

        else:
            if not self.backbone:
                raise ValueError("Backbone must be specified.")
            if not self.heads:
                raise ValueError("At least one head must be specified.")
            if self.additional_heads:
                warnings.warn("Ignoring specified additional heads.")
            if self.params:
                warnings.warn(
                    "Model-wise parameters ignored as no `predefined_model` specified."
                )

        return self

    # validation in json schema
    model_config = {
        "json_schema_extra": {
            "if": {"properties": {"predefined_model": {"type": "null"}}},
            "then": {
                "allOf": [
                    {"properties": {"backbone": {"type": "object"}}},
                    {"properties": {"heads": {"type": "array"}}},
                    {"required": ["backbone", "heads"]},
                ],
            },
        }
    }

    def _generate_predefined_model(
        self,
    ) -> Tuple[ModelModuleConfig, Optional[ModelModuleConfig], List[ModelHeadConfig]]:
        """Generates backbone, (optional) neck and heads configs based on specified predefined_model parameter

        Returns:
            Tuple[ModelModuleConfig, Optional[ModelModuleConfig], List[ModelHeadConfig]]: Backbone, (optional) neck and heads configs
        """
        if self.predefined_model.lower().startswith("yolov6"):
            version = self.predefined_model.lower().split("-")[-1]
            return self._generate_yolov6_model(version)
        else:
            raise ValueError(
                f"No predefined model with name `{self.predefined_model}`."
            )

    def _generate_yolov6_model(
        self, version: Literal["n", "t", "s"]
    ) -> Tuple[ModelModuleConfig, ModelModuleConfig, List[ModelHeadConfig]]:
        """Generates YoloV6 config based on specified version and model-wise parameters

        Args:
            version (Literal["n", "t", "s"]): YoloV6 version

        Returns:
            Tuple[ModelModuleConfig, ModelModuleConfig, List[ModelHeadConfig]]: Backbone, neck and heads configs
        """
        if version not in ["n", "t", "s"]:
            raise ValueError(f"Specified yolov6 version `{version}` not supported.")

        if version == "n":
            width_mul = 0.25
            iou_type = "siou"
        elif version == "s":
            width_mul = 0.375
            iou_type = "siou"
        else:
            width_mul = 0.5
            iou_type = "giou"

        backbone = ModelModuleConfig(
            name="EfficientRep",
            params={
                "channels_list": [64, 128, 256, 52, 1024],
                "num_repeats": [1, 6, 12, 18, 6],
                "depth_mul": self.params.get("depth_mul", 0.33),
                "width_mul": self.params.get("width_mul", width_mul),
            },
        )
        neck = ModelModuleConfig(
            name="RepPANNeck",
            params={
                "channels_list": [256, 128, 128, 256, 256, 512],
                "num_repeats": [12, 12, 12, 12],
                "num_heads": self.params.get("num_heads", 3),
                "depth_mul": self.params.get("depth_mul", 0.33),
                "width_mul": self.params.get("width_mul", width_mul),
            },
        )
        heads = [
            ModelHeadConfig(
                name="BboxYoloV6Loss",
                params={"num_heads": self.params.get("num_heads", 3)},
                loss=LossModuleConfig(
                    name="BboxYoloV6Loss", params={"iou_type": iou_type}
                ),
            )
        ]

        return backbone, neck, heads


# ----- TRAINER CONFIG ----
class TrainerConfig(BaseModel):
    accelerator: Literal["auto", "cpu", "gpu"] = "auto"
    devices: Union[int, List[int], str] = "auto"
    strategy: Literal["auto", "ddp"] = "auto"
    num_sanity_val_steps: int = 2
    profiler: Union[None, Literal["simple", "advanced"]] = None
    verbose: bool = False


# ----- LOGGER CONFIG ----
class LoggerConfig(BaseModel):
    project_name: Optional[str] = None
    project_id: Optional[str] = None
    run_name: Optional[str] = None
    run_id: Optional[str] = None
    save_directory: str = "output"
    is_tensorboard: bool = True
    is_wandb: bool = False
    wandb_entity: Optional[str] = None
    is_mlflow: bool = False
    logger_hyperparams: List[str] = ["train.epochs", "train.batch_size"]

    # Can also add extra validation based on Tracker class


# ----- DATASET CONFIG ----
class DatasetConfig(BaseModel):
    dataset_name: Optional[str] = None
    dataset_id: Optional[str] = None
    team_name: Optional[str] = None
    team_id: Optional[str] = None
    bucket_type: Literal[
        "BucketType.INTERNAL", "BucketType.EXTERNAL"
    ] = "BucketType.INTERNAL"
    bucket_storage: Literal[
        "BucketStorage.LOCAL",
        "BucketStorage.S3",
        "BucketStorage.GCS",
        "BucketStorage.AZURE_BLOB",
    ] = "BucketStorage.LOCAL"
    json_mode: bool = False
    train_view: str = "train"
    val_view: str = "val"
    test_view: str = "test"

    @model_validator(mode="after")
    def check_dataset_params(self) -> "DatasetConfig":
        if not self.dataset_name and not self.dataset_id:
            raise ValueError("Must provide either `dataset_name` or `dataset_id`.")
        return self

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
        }
    }


# ----- TRAIN CONFIG ----
class NormalizeAugmentationConfig(BaseModel):
    active: bool = True
    params: Dict[str, Any] = {}


class AugmentationConfig(BaseModel):
    name: str
    params: Dict[str, Any] = {}


class PreprocessingConfig(BaseModel):
    train_image_size: Annotated[
        List[int], Field(default=[256, 256], min_length=2, max_length=2)
    ]
    keep_aspect_ratio: bool = True
    train_rgb: bool = True
    normalize: NormalizeAugmentationConfig = NormalizeAugmentationConfig()
    augmentations: List[AugmentationConfig] = []

    @model_validator(mode="after")
    def check_normalize(self) -> "PreprocessingConfig":
        if self.normalize.active:
            self.augmentations.append(
                AugmentationConfig(name="Normalize", params=self.normalize.params)
            )
        return self


class ExportOnFinishConfig(BaseModel):
    active: bool = False
    override_upload_directory: bool = True


class CheckpointConfig(BaseModel):
    save_top_k: int = 3


class UploadCheckpointOnFinishConfig(BaseModel):
    active: bool = False
    upload_directory: Optional[str] = None

    @model_validator(mode="after")
    def check_upload_directory(self) -> "UploadCheckpointOnFinishConfig":
        if self.active and not self.upload_directory:
            raise ValueError("No `upload_directory` specified.")
        return self

    model_config = {
        "json_schema_extra": {
            "if": {"properties": {"active": {"const": True}}},
            "then": {
                "allOf": [
                    {"properties": {"upload_directory": {"type": "string"}}},
                    {
                        "required": [
                            "upload_directory",
                        ]
                    },
                ],
            },
        }
    }


class EarlyStoppingConfig(BaseModel):
    active: bool = True
    monitor: str = "val_loss/loss"
    mode: Literal["min", "max"] = "min"
    patience: int = 5
    verbose: bool = True


class CustomCallbackConfig(BaseModel):
    name: str
    params: Dict[str, Any] = {}


class CallbacksConfig(BaseModel):
    test_on_finish: bool = False
    use_device_stats_monitor: bool = False
    export_on_finish: ExportOnFinishConfig = ExportOnFinishConfig()
    checkpoint: CheckpointConfig = CheckpointConfig()
    upload_checkpoint_on_finish: UploadCheckpointOnFinishConfig = (
        UploadCheckpointOnFinishConfig()
    )
    early_stopping: EarlyStoppingConfig = EarlyStoppingConfig()
    custom_callbacks: List[CustomCallbackConfig] = []


class OptimizerConfig(BaseModel):
    name: str = "Adam"
    params: Dict[str, Any] = {}


class SchedulerConfig(BaseModel):
    name: str = "ConstantLR"
    params: Dict[str, Any] = {}


class OptimizersConfig(BaseModel):
    optimizer: OptimizerConfig = OptimizerConfig()
    scheduler: SchedulerConfig = SchedulerConfig()


class FreezeModulesConfig(BaseModel):
    backbone: bool = False
    neck: bool = False
    heads: List[bool] = [False]  # can also add validators but need ModelConfig context


class LossesConfig(BaseModel):
    log_sub_losses: bool = False
    weights: List[float] = [1]  # can also add validators but need ModelConfig context


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

    callbacks: CallbacksConfig = CallbacksConfig()

    optimizers: OptimizersConfig = OptimizersConfig()

    freeze_modules: FreezeModulesConfig = FreezeModulesConfig()

    losses: LossesConfig = LossesConfig()


# ----- INFERER CONFIG ----
class InfererConfig(BaseModel):
    dataset_view: str = "val"
    display: bool = True
    infer_save_directory: Optional[str] = None


# ----- EXPORTER CONFIG ----
class OnnxExportConfig(BaseModel):
    opset_version: int = 12
    dynamic_axes: Optional[Dict[str, Any]] = None


class OpenvinoExportConfig(BaseModel):
    active: bool = False


class BlobconverterExportConfig(BaseModel):
    active: bool = False
    shaves: int = 6


class UploadExportConfig(BaseModel):
    active: bool = False
    upload_directory: Optional[str] = None

    @model_validator(mode="after")
    def check_upload_directory(self) -> "UploadExportConfig":
        if self.active and not self.upload_directory:
            raise ValueError("No `upload_directory` specified.")
        return self

    model_config = {
        "json_schema_extra": {
            "if": {"properties": {"active": {"const": True}}},
            "then": {
                "allOf": [
                    {"properties": {"upload_directory": {"type": "string"}}},
                    {
                        "required": [
                            "upload_directory",
                        ]
                    },
                ],
            },
        }
    }


class ExporterConfig(BaseModel):
    export_weights: Optional[str] = None
    export_save_directory: str = "output_export"
    export_image_size: Annotated[
        List[int], Field(default=[256, 256], min_length=2, max_length=2)
    ]
    export_model_name: str = "model"
    data_type: Literal["INT8", "FP16", "FP32"] = "FP16"
    reverse_input_channels: bool = True
    scale_values: Union[List[float], float] = [58.395, 57.120, 57.375]
    mean_values: Union[List[float], float] = [123.675, 116.28, 103.53]
    onnx: OnnxExportConfig = OnnxExportConfig()
    openvino: OpenvinoExportConfig = OpenvinoExportConfig()
    blobconverter: BlobconverterExportConfig = BlobconverterExportConfig()
    upload: UploadExportConfig = UploadExportConfig()


# ----- TUNER CONFIG ----
class StorageConfig(BaseModel):
    active: bool = True
    type: Literal["local", "remote"] = "local"


class TunerConfig(BaseModel):
    study_name: str = "test-study"
    use_pruner: bool = True
    n_trials: int = 3
    timeout: Optional[int] = None
    storage: StorageConfig = StorageConfig()
    params: Dict[str, List[str | int | float | bool]] = {}


# ---- BASE CONFIG ----
class Config(BaseModel):
    model: ModelConfig
    trainer: TrainerConfig = TrainerConfig()
    logger: LoggerConfig = LoggerConfig()
    dataset: DatasetConfig
    train: TrainConfig = TrainConfig()
    inferer: InfererConfig = InfererConfig()
    exporter: ExporterConfig = ExporterConfig()
    tuner: TunerConfig = TunerConfig()


# import yaml
# with open("test.yaml", "r") as f:
#     cfg = yaml.load(f, Loader=yaml.SafeLoader)

# # print(cfg["dataset"])
# config = Config(**cfg)
# print(config.model)

# print(config.model_dump_json(indent=2))

# import json
# schema = json.dumps(Config.model_json_schema(), indent=2)
# print(schema)

# import jsonschema
# jsonschema.validate(config.model_dump_json(), schema)
