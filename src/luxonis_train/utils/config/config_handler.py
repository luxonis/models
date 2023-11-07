import warnings
from typing import Any, Dict, List

from .config import *
from luxonis_train.utils.registry import HEADS
from luxonis_ml.data import LuxonisDataset, LabelType
from luxonis_ml.utils import ConfigHandler as BaseConfigHandler


class ConfigHandler(BaseConfigHandler):
    def __new__(
        cls, cfg: str | Dict[str, Any] | None = None, cfg_cls: type | None = None
    ):
        return super().__new__(cls, cfg=cfg, cfg_cls=Config)

    def _validate(self) -> None:
        """Performs custom validation of the config"""
        self._validate_dataset_classes()
        self._validate_after()

    def _validate_dataset_classes(self) -> None:
        """Validates config to used datasets, overrides n_classes if needed"""
        with LuxonisDataset(
            dataset_name=self.get("dataset.dataset_name"),
            team_id=self.get("dataset.team_id"),
            dataset_id=self.get("dataset.dataset_id"),
            bucket_type=self.get("dataset.bucket_type"),
            bucket_storage=self.get("dataset.bucket_storage"),
        ) as dataset:
            classes, classes_by_task = dataset.get_classes()

            if not classes:
                raise ValueError("Provided dataset doesn't have any classes.")

            for head in self.get("model.heads"):
                label_type = get_head_label_types(head.name)[0]
                dataset_n_classes = len(classes_by_task[label_type.value])

                if "n_classes" in head.params:
                    if head.params.get("n_classes") != dataset_n_classes:
                        raise ValueError(
                            f"Number of classes in config ({head.params.get('n_classes')}) doesn't match number of "
                            f"classes in dataset ({dataset_n_classes}) for `{head.name}`."
                        )
                else:
                    warnings.warn(
                        f"Inheriting 'n_classes' parameter for `{head.name}` from dataset. Setting it to {dataset_n_classes}."
                    )
                    head.params["n_classes"] = dataset_n_classes

    def _validate_after(self) -> None:
        """Performs any additional validation on the top level after the fact"""
        # validate KeypointHead anchors
        for head in self.get("model.heads"):
            if head.name == "KeypointBboxHead" and head.params.get("anchors") == None:
                warnings.warn("Generating")
                anchors = self._autogenerate_anchors(head)
                head.params["anchors"] = anchors

        n_heads = len(self.get("model.heads"))
        # validate freeze_modules
        if len(self.get("train.freeze_modules.heads")) != n_heads:
            self._data.train.freeze_modules.heads = clip_or_fill_list(
                self.get("train.freeze_modules.heads"), n_heads, False
            )
        # validate loss weights
        if len(self.get("train.losses.weights")) != n_heads:
            self._data.train.losses.weights = clip_or_fill_list(
                self.get("train.losses.weights"), n_heads, 1
            )

    def _autogenerate_anchors(self, head: ModelHeadConfig) -> List[List[float]]:
        """Automatically generates anchors for the provided dataset

        Args:
            head (ModelHeadConfig): Config of the head where anchors will be used

        Returns:
            List[List[float]]: List of anchors in [-1,6] format
        """
        from torch.utils.data import DataLoader, ValAugmentations, LuxonisLoader

        from luxonis_train.utils.boxutils import anchors_from_dataset
        from luxonis_train.utils.loader import collate_fn

        with LuxonisDataset(
            dataset_name=self.get("dataset.dataset_name"),
            team_id=self.get("dataset.team_id"),
            dataset_id=self.get("dataset.dataset_id"),
            bucket_type=self.get("dataset.bucket_type"),
            bucket_storage=self.get("dataset.bucket_storage"),
        ) as dataset:
            val_augmentations = ValAugmentations(
                image_size=self.get("train.preprocessing.train_image_size"),
                augmentations=[{"name": "Normalize", "params": {}}],
                train_rgb=self.get("train.preprocessing.train_rgb"),
                keep_aspect_ratio=self.get("train.preprocessing.keep_aspect_ratio"),
            )
            loader = LuxonisLoader(
                dataset,
                view=self.get("dataset.train_view"),
                augmentations=val_augmentations,
                mode="json" if self.get("dataset.json_mode") else "fiftyone",
            )
            pytorch_loader = DataLoader(
                loader,
                batch_size=self.get("train.batch_size"),
                num_workers=self.get("train.num_workers"),
                collate_fn=collate_fn,
            )
            num_heads = head.params.get("num_heads", 3)
            proposed_anchors = anchors_from_dataset(
                pytorch_loader, n_anchors=num_heads * 3
            )
            return proposed_anchors.reshape(-1, 6).tolist()


def get_head_label_types(head_str: str) -> List[LabelType]:
    """Returns all label types defined as head class attributes"""
    return HEADS.get(head_str).label_types


def clip_or_fill_list(
    input_list: List[Any], target_len: int, fill_value: Any
) -> List[Any]:
    """Clips of fills the list inplace so its length is target_len

    Args:
        input_list (List[Any]): List to clip or fill
        target_len (int): Desired target length of the list
        fill_value (Any): Value used for fill

    Returns:
        List[Any]: List of length target_len
    """
    if len(input_list) > target_len:
        input_list = input_list[:target_len]
    elif len(input_list) < target_len:
        input_list.extend([fill_value] * (target_len - len(input_list)))
    return input_list
