import math
import torch


def make_divisible(x: int, divisor: int):
    """Upward revision the value x to make it evenly divisible by the divisor."""
    return math.ceil(x / divisor) * divisor


def dummy_input_run(
    module: torch.nn.Module, input_shape: list, multi_input: bool = False
):
    """Runs dummy input through the module"""
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
