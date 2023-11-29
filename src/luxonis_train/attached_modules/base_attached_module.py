from abc import ABC
from typing import Generic

from pydantic import ValidationError
from torch import Size, Tensor, nn
from typing_extensions import TypeVarTuple, Unpack

from luxonis_train.utils.general import DatasetMetadata, validate_packet
from luxonis_train.utils.registry import AutoRegisterMeta
from luxonis_train.utils.types import (
    BaseProtocol,
    IncompatibleException,
    Kwargs,
    Labels,
    LabelType,
    Packet,
)

Ts = TypeVarTuple("Ts")


class BaseAttachedModule(
    nn.Module, Generic[Unpack[Ts]], ABC, metaclass=AutoRegisterMeta, register=False
):
    """Base class for all modules that are attached to a `LuxonisNode`.

    Attached modules include losses, metrics and visualizers.

    This class contains a default implementation of `prepare` method, which
    should be sufficient for most simple cases. More complex modules should
    override the `prepare` method.

    Metaclass Args:
        register (bool): Determines whether or not to register this class.
          Should be set to False in abstract classes to prevent them
          from being registered.
        registry (Registry): The registry to which the subclasses should be added.
          For most times should not be specified in concrete classes.

    Attributes:
        required_labels (list[LabelType]): List of labels required by this model.
        protocol (type[BaseProtocol]): Schema for validating inputs to the module.
        node_attributes (NodeAttributes): Attributes of the node that
          this module is attached to.

    Interface:
        prepare(outputs: Packet[Tensor], labels: Labels
        ) -> tuple[Unpack[Ts]]:
          Prepares the outputs and labels before passing them to following methods.
          For example with `forward`, it would allow for the following call:
          `forward(*prepare(outputs, labels))`.
    """

    class NodeAttributes(BaseProtocol):
        """Contains attributes of the node that this module is attached to.

        Common attributes for all nodes are defined here.

        A subclass of `LuxonisAttachedModule` can define its own `NodeAttributes`
        class. In that case, the subclass' `NodeAttributes` should inherit from
        `LuxonisAttachedModule.NodeAttributes` and only define additional attributes.

        Definition of custom `NodeAttributes` is not required, but if provided,
        the passed node attributes will be validated against the `NodeAttributes`
        schema.


        Attributes:
            original_in_shape (Size): Shape of the input to the entire model.
            dataset_metadata (DatasetMetadata): Metadata of the dataset.
        """

        original_in_shape: Size
        dataset_metadata: DatasetMetadata

    def __init__(
        self,
        *,
        node_attributes: Kwargs | NodeAttributes | None = None,
        protocol: type[BaseProtocol] | None = None,
        required_labels: list[LabelType] | None = None,
    ):
        """Initializes the module.

        Args:
            node_attributes (Kwargs, optional): Attributes of the node that this module is attached to.
              Defaults to None.
            protocol (type[BaseProtocol], optional): Protocol that the node attributes must conform to.
              Defaults to None.
            required_labels (list[LabelType], optional): List of labels that this module requires.
              Defaults to None.
        """
        self.required_labels = required_labels or []
        self.protocol = protocol
        try:
            if isinstance(node_attributes, dict):
                self._node_attributes = self.NodeAttributes(**node_attributes)
            else:
                self._node_attributes = node_attributes
        except ValidationError as e:
            raise IncompatibleException.from_validation_error(
                e, self.__class__.__name__
            ) from e
        self._epoch = 0
        super().__init__()

    @property
    def node_attributes(self) -> NodeAttributes:
        """Attributes of the node that this module is attached to.

        Returns:
            NodeAttributes: Attributes of the node that this module is attached to.
        """
        if self._node_attributes is None:
            raise RuntimeError(
                "Attempt to access `node_attributes`, but they were not "
                "provided during initialization."
            )
        return self._node_attributes

    def prepare(self, inputs: Packet[Tensor], labels: Labels) -> tuple[Unpack[Ts]]:
        """Prepares node outputs for the forward pass of the module.

        This default implementation selects the output and label based on
        `required_labels` attribute. If not set, then it returns the first
        matching output and label.
        That is the first pair of outputs and labels that have the same type.
        For more complex modules this method should be overridden.

        Args:
            inputs (Packet[Tensor]): Output from the node, inputs to the attached module.
            labels (Labels): Labels from the dataset.

        Returns:
            tuple[PredictionType, TargetType]: Prepared inputs. Should allow the
              following usage with the `forward` method:
                `loss.forward(*loss.prepare(outputs, labels))`

        Raises:
            IncompatibleException: If the inputs are not compatible with the module.
        """
        if len(self.required_labels) > 1:
            raise NotImplementedError(
                "This module requires multiple labels, the default `prepare` "
                "implementation does not support this."
            )
        if not self.required_labels:
            if "boxes" in inputs and LabelType.BOUNDINGBOX in labels:
                return inputs["boxes"], labels[LabelType.BOUNDINGBOX]  # type: ignore
            if "classes" in inputs and LabelType.CLASSIFICATION in labels:
                return inputs["classes"], labels[LabelType.CLASSIFICATION]  # type: ignore
            if "keypoints" in inputs and LabelType.KEYPOINT in labels:
                return inputs["keypoints"], labels[LabelType.KEYPOINT]  # type: ignore
            if "segmentation" in inputs and LabelType.SEGMENTATION in labels:
                return inputs["segmentation"][0], labels[LabelType.SEGMENTATION]  # type: ignore
            raise IncompatibleException(
                f"No matching labels and outputs found for {self.__class__.__name__}"
            )
        label_type = self.required_labels[0]
        return inputs[label_type.value], labels[label_type]  # type: ignore

    def validate(self, inputs: Packet[Tensor], labels: Labels) -> None:
        """Validates that the inputs and labels are compatible with the module.

        Args:
            inputs (Packet[Tensor]): Inputs to the module.
            labels (Labels): Labels from the dataset.
        """
        for label in self.required_labels:
            if label not in labels:
                raise IncompatibleException.from_missing_label(
                    label, list(labels.keys()), self.__class__.__name__
                )

        if self.protocol is not None:
            try:
                validate_packet(inputs, self.protocol)
            except ValidationError as e:
                raise IncompatibleException.from_validation_error(
                    e, self.__class__.__name__
                ) from e
