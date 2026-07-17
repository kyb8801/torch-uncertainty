import torch
from torch import Tensor


def check_interval_shapes(lower: Tensor, upper: Tensor, target: Tensor | None = None) -> None:
    """Check the shape and values of a tensor of intervals or quantiles.

    All lower values should be lower than the upper values and both tensors should have the same
    size.
    """
    if lower.shape != upper.shape:
        raise ValueError(
            f"Expected `lower` and `upper` to have the same shape, got {lower.shape=} and "
            f"{upper.shape=}."
        )
    if target is not None and target.shape != lower.shape:
        raise ValueError(
            f"Expected `target` to have the same shape as the interval bounds, got "
            f"{target.shape=} and {lower.shape=}."
        )
    if torch.any(lower > upper):
        raise ValueError("Expected `lower` to be less than or equal to `upper` elementwise.")
