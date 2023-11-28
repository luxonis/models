import math
import warnings
from typing import Generator, TypeVar

from luxonis_ml.data import LuxonisDataset
from pydantic import BaseModel
from torch import Size, Tensor

from luxonis_train.utils.types import Packet


# TODO: could be moved to luxonis-ml
class DatasetMetadata(BaseModel):
    """Metadata about the dataset.

    Attributes:
        n_classes (int): Number of classes in the dataset.
        n_keypoints (int): Number of keypoints in the dataset.
        class_names (list[str]): Names of the classes in the dataset.
    """

    n_classes: int = 0
    n_keypoints: int = 0
    class_names: list[str] = []

    @classmethod
    def from_dataset(cls, dataset: LuxonisDataset) -> "DatasetMetadata":
        """Creates a `DatasetMetadata` object from a `LuxonisDataset`.

        Args:
            dataset (LuxonisDataset): Dataset to create the metadata from.

        Returns:
            DatasetMetadata: Metadata about the dataset.
        """
        class_names = dataset.get_classes()[0]
        # TODO: this is not optimal
        skeletons = dataset.get_skeletons()
        return cls(
            n_classes=len(class_names),
            n_keypoints=len(skeletons[class_names[0]]["labels"]),
            class_names=class_names,
        )


# TEST:
def make_divisible(x: int | float, divisor: int) -> int:
    """Upward revision the value x to make it evenly divisible by the divisor."""
    return math.ceil(x / divisor) * divisor


# TEST:
def infer_upscale_factor(
    in_height: int, orig_height: int, strict: bool = True, warn: bool = True
) -> int:
    """Infer the upscale factor from the input height and original height."""
    num_up = math.log2(orig_height) - math.log2(in_height)
    if num_up.is_integer():
        return int(num_up)
    elif not strict:
        if warn:
            warnings.warn(
                f"Upscale factor is not an integer: {num_up}. "
                "Output shape will not be the same as input shape."
            )
        return round(num_up)
    else:
        raise ValueError(
            f"Upscale factor is not an integer: {num_up}. "
            "Output shape will not be the same as input shape."
        )


def get_shape_packet(packet: Packet[Tensor]) -> Packet[Size]:
    shape_packet: Packet[Size] = {}
    for name, value in packet.items():
        shape_packet[name] = [x.shape for x in value]
    return shape_packet


# TEST:
def is_acyclic(graph: dict[str, list[str]]) -> bool:
    """Tests if graph is acyclic.

    Args:
        graph (dict[str, list[str]]): Graph in a format of a dictionary
          of predecessors. Keys are node names, values are inputs to the
          node (list of node names).

    Returns:
        bool: True if graph is acyclic, False otherwise.
    """
    graph = graph.copy()

    def dfs(node: str, visited: set[str], recursion_stack: set[str]):
        visited.add(node)
        recursion_stack.add(node)

        for predecessor in graph.get(node, []):
            if predecessor in recursion_stack:
                return True
            if predecessor not in visited:
                if dfs(predecessor, visited, recursion_stack):
                    return True

        recursion_stack.remove(node)
        return False

    visited: set[str] = set()
    recursion_stack: set[str] = set()

    for node in graph.keys():
        if node not in visited:
            if dfs(node, visited, recursion_stack):
                return False

    return True


def validate_packet(data: Packet, protocol: type[BaseModel]) -> Packet:
    return protocol(**data).model_dump()


T = TypeVar("T")


def traverse_graph(
    graph: dict[str, list[str]], nodes: dict[str, T]
) -> Generator[tuple[str, T, list[str], set[str]], None, None]:
    unprocessed_nodes = set(nodes.keys())
    processed = set()

    while unprocessed_nodes:
        unprocessed_nodes_copy = unprocessed_nodes.copy()
        for node_name in unprocessed_nodes_copy:
            node_dependencies = graph[node_name]
            if not node_dependencies or all(
                dependency in processed for dependency in node_dependencies
            ):
                yield node_name, nodes[node_name], node_dependencies, unprocessed_nodes
                processed.add(node_name)
                unprocessed_nodes.remove(node_name)

        if unprocessed_nodes_copy == unprocessed_nodes:
            raise RuntimeError(
                "Malformed graph. "
                "Please check that all nodes are connected in a directed acyclic graph."
            )
