from collections.abc import Iterator

import torch
from torch import Tensor


def binary_images(preds: Tensor, target: Tensor) -> Iterator[tuple[Tensor, Tensor]]:
    """Yield flattened prediction/target pairs, one image at a time."""
    if preds.shape != target.shape:
        raise ValueError(
            "Expected `preds` and `target` to have the same shape, but got "
            f"{preds.shape} and {target.shape}."
        )
    if preds.ndim == 0:
        raise ValueError("Expected `preds` and `target` to have at least one dimension.")

    if preds.ndim == 1:
        preds = preds.unsqueeze(0)
        target = target.unsqueeze(0)
    else:
        preds = preds.flatten(start_dim=1)
        target = target.flatten(start_dim=1)

    yield from zip(preds, target, strict=True)


def binary_target_has_classes(
    target: Tensor,
    *,
    pos_label: int = 1,
    ignore_index: int | None = None,
    require_negative: bool = True,
) -> bool:
    """Check whether an image contains the classes required by a binary metric."""
    if ignore_index is not None:
        target = target[target != ignore_index]
    if target.numel() == 0:
        return False

    positive = target == pos_label
    has_positive = torch.any(positive).item()
    if not require_negative:
        return has_positive
    return has_positive and torch.any(~positive).item()
