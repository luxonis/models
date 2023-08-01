import torch
import torch.nn as nn
import cv2
from typing import List, Union
from torchvision.utils import draw_segmentation_masks

from luxonis_ml.loader import LabelType
from luxonis_train.utils.visualization import (
    torch_img_to_numpy,
    numpy_to_torch_img,
    seg_output_to_bool,
)
from luxonis_train.utils.constants import HeadType


class BaseHead(nn.Module):
    head_types: List[HeadType] = []
    label_types: List[LabelType] = []

    def __init__(
        self,
        n_classes: int,
        prev_out_shapes: list,
        original_in_shape: list,
        attach_index: int = -1,
    ):
        """Base abstract head class from which all other heads are created

        Args:
            n_classes (int): Number of classes
            prev_out_shapes (list): List of shapes of previous outputs
            original_in_shape (list): Original input shape to the model
            attach_index (int, optional): Index of previous output that the head attaches to. Defaults to -1.
        """
        super().__init__()

        self.n_classes: int = n_classes
        self.attach_index: int = attach_index
        self.prev_out_shapes: list = prev_out_shapes
        self.original_in_shape: list = original_in_shape

    def get_name(self, idx: int):
        """Generate a string head name based on class name and passed index"""
        return f"{self.__class__.__name__}_{idx}"

    def check_annotations(self, label_dict: dict):
        """Checks if all required annotations are present in label dictionary

        Args:
            label_dict (dict): Dictionary with labels

        Raises:
            KeyError: Required label type missing in annotations
        """
        for label_type in self.label_types:
            if label_type not in label_dict:
                raise KeyError(
                    f"Required label type `{label_type}` missing in annotations."
                )

    def forward(self, x):
        """torch.nn.Module forward method"""
        raise NotImplementedError

    def postprocess_for_loss(
        self, output: Union[tuple, torch.Tensor], label_dict: dict
    ):
        """Performs postprocessing on output and label for loss function input

        Args:
            output (torch.Tensor): Output of current head
            label_dict (dict): Dict of all present labels in the dataset

        Returns:
            output: Transformed output ready for loss prediction
            label: Transformed label ready for loss target
        """
        raise NotImplementedError

    def postprocess_for_metric(
        self, output: Union[tuple, torch.Tensor], label_dict: dict
    ):
        """Performs postprocessing on output and label for metric comput input

        Args:
            output (torch.Tensor): Output of current head
            label_dict (dict): Dict of all present labels in the dataset

        Returns:
            output: Transformed output ready for metric computation
            label: Transformed label ready for metric computation
            metric_mapping (Union[Dict[str, int], None]): Mapping between metric name and index \
                of returned Tuple. Used for multi-task heads. If not needed set to None.
        """
        raise NotImplementedError

    def draw_output_to_img(self, img: torch.Tensor, output: torch.Tensor, idx: int):
        """Draws model output to an img

        Args:
            img (torch.Tensor): Current image
            output (torch.Tensor): Output of the current head
            idx (int): Index of the image in the batch

        Returns:
            img (torch.Tensor): Output img with correct output drawn on
        """
        raise NotImplementedError

    def get_output_names(self, idx: int):
        """Get output names used for export

        Args:
            idx (int): Head index in the model

        Returns:
            output_name (Union[str, List[str]]): Either output name (string) or list of strings if head has multiple outputs.
        """
        raise NotImplementedError

    def to_deploy(self):
        """Makes changes needed to prepare head for deploy"""
        pass


class BaseClassificationHead(BaseHead):
    head_types: List[HeadType] = [HeadType.CLASSIFICATION]
    label_types: List[LabelType] = [LabelType.CLASSIFICATION]

    def __init__(
        self,
        n_classes: int,
        prev_out_shapes: list,
        original_in_shape: list,
        attach_index: int = -1,
        main_metric: str = "f1",
    ):
        """Base head for classification tasks

        Args:
            n_classes (int): Number of classes
            prev_out_shapes (list): List of shapes of previous outputs
            original_in_shape (list): Original input shape to the model
            attach_index (int, optional): Index of previous output that the head attaches to. Defaults to -1.
            main_metric (str, optional): Name of the main metric which is used for tracking training process. Defaults to "f1".
        """
        super().__init__(
            n_classes=n_classes,
            prev_out_shapes=prev_out_shapes,
            original_in_shape=original_in_shape,
            attach_index=attach_index,
        )

        self.main_metric: str = main_metric

    def postprocess_for_loss(self, output: torch.Tensor, label_dict: dict):
        label = label_dict[self.label_types[0]]
        return output, label

    def postprocess_for_metric(self, output: torch.Tensor, label_dict: dict):
        label = label_dict[self.label_types[0]]
        if self.n_classes != 1:
            label = torch.argmax(label, dim=1)
            output = torch.argmax(output, dim=1)
        return output, label, None

    def draw_output_to_img(self, img: torch.Tensor, output: torch.Tensor, idx: int):
        # convert image to cv2 to add text and then convert back to torch img
        # TODO: find if there is a better way to add text to torch img
        img = torch_img_to_numpy(img)
        curr_class = output[idx].argmax()
        img = cv2.putText(
            img,
            f"{curr_class}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        img = numpy_to_torch_img(img)
        return img

    def get_output_names(self, idx: int):
        return f"output{idx}"


class BaseMultiLabelClassificationHead(BaseHead):
    head_types: List[HeadType] = [HeadType.MULTI_LABEL_CLASSIFICATION]
    label_types: List[LabelType] = [LabelType.CLASSIFICATION]

    def __init__(
        self,
        n_classes: int,
        prev_out_shapes: list,
        original_in_shape: list,
        attach_index: int = -1,
        main_metric: str = "f1",
    ):
        """Base head for multi-label classification tasks

        Args:
            n_classes (int): Number of classes
            prev_out_shapes (list): List of shapes of previous outputs
            original_in_shape (list): Original input shape to the model
            attach_index (int, optional): Index of previous output that the head attaches to. Defaults to -1.
            main_metric (str, optional): Name of the main metric which is used for tracking training process. Defaults to "f1".
        """
        super().__init__(
            n_classes=n_classes,
            prev_out_shapes=prev_out_shapes,
            original_in_shape=original_in_shape,
            attach_index=attach_index,
        )

        self.main_metric: str = main_metric

    def postprocess_for_loss(self, output: torch.Tensor, label_dict: dict):
        label = label_dict[self.label_types[0]]
        return output, label

    def postprocess_for_metric(self, output: torch.Tensor, label_dict: dict):
        label = label_dict[self.label_types[0]]
        return output, label, None

    def draw_output_to_img(self, img: torch.Tensor, output: torch.Tensor, idx: int):
        # convert image to cv2 to add text and then convert back to torch img
        # TODO: find if there is a better way to add text to torch img
        img = torch_img_to_numpy(img)
        indices = torch.nonzero(output[idx]).flatten().tolist()
        curr_label_str = ",".join(str(i) for i in indices)
        img = cv2.putText(
            img,
            f"{curr_label_str}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        img = numpy_to_torch_img(img)
        return img

    def get_output_names(self, idx: int):
        return f"output{idx}"


class BaseSegmentationHead(BaseHead):
    head_types: List[HeadType] = [HeadType.SEMANTIC_SEGMENTATION]
    label_types: List[LabelType] = [LabelType.SEGMENTATION]

    def __init__(
        self,
        n_classes: int,
        prev_out_shapes: list,
        original_in_shape: list,
        attach_index: int = -1,
        main_metric: str = "mIoU",
    ):
        """Base head for segmentation tasks

        Args:
            n_classes (int): Number of classes
            prev_out_shapes (list): List of shapes of previous outputs
            original_in_shape (list): Original input shape to the model
            attach_index (int, optional): Index of previous output that the head attaches to. Defaults to -1.
            main_metric (str, optional): Name of the main metric which is used for tracking training process. Defaults to "mIoU".
        """
        super().__init__(
            n_classes=n_classes,
            prev_out_shapes=prev_out_shapes,
            original_in_shape=original_in_shape,
            attach_index=attach_index,
        )

        self.main_metric: str = main_metric

    def postprocess_for_loss(self, output: torch.Tensor, label_dict: dict):
        label = label_dict[self.label_types[0]]
        return output, label

    def postprocess_for_metric(self, output: torch.Tensor, label_dict: dict):
        label = label_dict[self.label_types[0]]
        if self.n_classes != 1:
            labels = torch.argmax(labels, dim=1)
        return output, label, None

    def draw_output_to_img(self, img: torch.Tensor, output: torch.Tensor, idx: int):
        masks = seg_output_to_bool(output[idx])
        # NOTE: we have to push everything to cpu manually before draw_segmentation_masks (torchvision bug?)
        masks = masks.cpu()
        img = img.cpu()
        img = draw_segmentation_masks(img, masks, alpha=0.4)
        return img

    def get_output_names(self, idx: int):
        # TODO: should we add index if there are multiple segmentation heads?
        return f"segmentation"


class BaseObjectDetection(BaseHead):
    head_types: List[HeadType] = [HeadType.OBJECT_DETECTION]
    label_types: List[LabelType] = [LabelType.BOUNDINGBOX]

    def __init__(
        self,
        n_classes: int,
        prev_out_shapes: list,
        original_in_shape: list,
        attach_index: int = -1,
        main_metric: str = "map",
    ):
        """Base head for object detection tasks

        Args:
            n_classes (int): Number of classes
            prev_out_shapes (list): List of shapes of previous outputs
            original_in_shape (list): Original input shape to the model
            attach_index (int, optional): Index of previous output that the head attaches to. Defaults to -1.
            main_metric (str, optional): Name of the main metric which is used for tracking training process. Defaults to "map".
        """
        super().__init__(
            n_classes=n_classes,
            prev_out_shapes=prev_out_shapes,
            original_in_shape=original_in_shape,
            attach_index=attach_index,
        )

        self.main_metric: str = main_metric

    # NOTE: other methods not implemented as they are usually very head specific so we implement them per head


class BaseKeypointDetection(BaseHead):
    head_types: List[HeadType] = [HeadType.KEYPOINT_DETECTION]
    label_types: List[LabelType] = [LabelType.KEYPOINT]

    def __init__(
        self,
        n_classes: int,
        prev_out_shapes: list,
        original_in_shape: list,
        attach_index: int = -1,
        main_metric: str = "oks",
    ):
        """Base head for keypoint detection tasks

        Args:
            n_classes (int): Number of classes (usually refers to number of keypoints)
            prev_out_shapes (list): List of shapes of previous outputs
            original_in_shape (list): Original input shape to the model
            attach_index (int, optional): Index of previous output that the head attaches to. Defaults to -1.
            main_metric (str, optional): Name of the main metric which is used for tracking training process. Defaults to "oks".
        """
        super().__init__(
            n_classes=n_classes,
            prev_out_shapes=prev_out_shapes,
            original_in_shape=original_in_shape,
            attach_index=attach_index,
        )

        self.main_metric: str = main_metric

    # NOTE: other methods not implemented as they are usually very head specific so we implement them per head
