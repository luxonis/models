import math
import torch
from typing import List, Union


def make_divisible(x: int, divisor: int) -> int:
    """Upward revision the value x to make it evenly divisible by the divisor"""
    return math.ceil(x / divisor) * divisor


def dummy_input_run(
    module: torch.nn.Module,
    input_shape: List[Union[int, List[int]]],
    multi_input: bool = False,
) -> List[List[int]]:
    """Runs dummy input through the module and return output shapes

    Args:
        module (torch.nn.Module): Torch module
        input_shape (List[int]): Shape of the input
        multi_input (bool, optional): Whether module requires multiple inputs.
            Defaults to False.

    Returns:
        List[List[int]]: Shapes of each module output
    """
    module.eval()
    if multi_input:
        input = [torch.zeros(i) for i in input_shape]
    else:
        input = torch.zeros(input_shape)

    out = module(input)
    module.train()
    if isinstance(out, list):
        shapes = []
        for o in out:
            shapes.append(list(o.shape))
        return shapes
    else:
        return [list(out.shape)]
