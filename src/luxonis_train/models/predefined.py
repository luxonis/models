from typing import Any

from luxonis_train.utils.config import ModelNodeConfig
from luxonis_train.utils.registry import MODELS


@MODELS.register_module(name="SegmentationModel")
def segmentation_model(
    backbone: str = "MicroNet", params: dict[str, Any] | None = None
) -> list[ModelNodeConfig]:
    params = params or {}
    return [
        ModelNodeConfig(
            name=backbone,
            override_name="segmentation_backbone",
            params=params.get("backbone", {}),
        ),
        ModelNodeConfig(
            name="SegmentationHead",
            override_name="segmentation_head",
            inputs=["segmentation_backbone"],
            params=params.get("head", {}),
        ),
    ]


@MODELS.register_module(name="DetectionModel")
def detection_model(
    use_neck: bool = True, params: dict[str, Any] | None = None
) -> list[ModelNodeConfig]:
    params = params or {}
    nodes = [
        ModelNodeConfig(
            name="EfficientRep",
            override_name="detection_backbone",
            params=params.get("backbone", {}),
        ),
    ]
    if use_neck:
        nodes.append(
            ModelNodeConfig(
                name="RepPANNeck",
                override_name="detection_neck",
                inputs=["detection_backbone"],
                params=params.get("neck", {}),
            )
        )

    nodes.append(
        ModelNodeConfig(
            name="EfficientBBoxHead",
            inputs=["detection_head"],
            params=params.get("head", {}),
        )
    )
    return nodes


@MODELS.register_module(name="ImplicitKeypointModel")
def implicit_keypoint_model(
    use_neck: bool = True, params: dict[str, Any] | None = None
) -> list[ModelNodeConfig]:
    params = params or {}
    nodes = [
        ModelNodeConfig(
            name="EfficientRep",
            override_name="keypoint_backbone",
            params=params.get("backbone", {}),
        ),
    ]
    if use_neck:
        nodes.append(
            ModelNodeConfig(
                name="RepPANNeck",
                override_name="keypoint_neck",
                inputs=["keypoint_backbone"],
                params=params.get("neck", {}),
            )
        )

    nodes.append(
        ModelNodeConfig(
            name="ImplicitKeypointHead",
            inputs=["keypoint_head"],
            params=params.get("head", {}),
        )
    )
    return nodes
