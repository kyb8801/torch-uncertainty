from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.functional.classification.calibration_error import (
    _binary_calibration_error_arg_validation,
    _binary_calibration_error_tensor_validation,
    _ce_compute,
    _multiclass_calibration_error_arg_validation,
    _multiclass_calibration_error_tensor_validation,
)
from torchmetrics.utilities.data import dim_zero_cat
from torchmetrics.utilities.enums import ClassificationTaskNoMultilabel


def _patched_binary_calibration_error_arg_validation(
    patch_size: int,
    n_bins: int,
    norm: Literal["l1", "l2", "max"],
    ignore_index: int | None,
) -> None:
    _binary_calibration_error_arg_validation(n_bins, norm, ignore_index)
    if not isinstance(patch_size, int) or patch_size < 1:
        raise ValueError(
            f"Expected argument `patch_size` to be a positive integer, but got {patch_size}"
        )


def _patched_multiclass_calibration_error_arg_validation(
    patch_size: int,
    num_classses: int,
    n_bins: int,
    norm: Literal["l1", "l2", "max"],
    ignore_index: int | None,
) -> None:
    _multiclass_calibration_error_arg_validation(num_classses, n_bins, norm, ignore_index)
    if not isinstance(patch_size, int) or patch_size < 1:
        raise ValueError(
            f"Expected argument `patch_size` to be a positive integer, but got {patch_size}"
        )


def _pool_patches(
    conf: Tensor,
    acc: Tensor,
    valid: Tensor,
    patch_size: int | list[int] | tuple[int, int],
) -> tuple[Tensor, Tensor]:
    """Pool per-pixel confidence and accuracy over non-overlapping spatial patches.

    Args:
        conf: Per-pixel confidence, shape ``(B, H, W)``.
        acc: Per-pixel accuracy, shape ``(B, H, W)``.
        valid: Binary mask of valid (non-ignored) pixels, shape ``(B, H, W)``.
        patch_size: Side length of each square patch.

    Returns:
        Tuple of ``(patch_confidences, patch_accuracies)`` with one entry per
        patch that contains at least one valid pixel.
    """
    ps = patch_size
    n_pixels = ps * ps

    # avg_pool2d requires (B, C, H, W); multiply by n_pixels to recover sums
    conf_sum = (
        F.avg_pool2d(
            (conf * valid).unsqueeze(1), kernel_size=ps, stride=ps, count_include_pad=False
        )
        * n_pixels
    )
    acc_sum = (
        F.avg_pool2d((acc * valid).unsqueeze(1), kernel_size=ps, stride=ps, count_include_pad=False)
        * n_pixels
    )
    count = (
        F.avg_pool2d(valid.unsqueeze(1), kernel_size=ps, stride=ps, count_include_pad=False)
        * n_pixels
    )

    conf_sum = conf_sum.flatten()
    acc_sum = acc_sum.flatten()
    count = count.flatten()

    valid_patches = count > 0
    patch_conf = conf_sum[valid_patches] / count[valid_patches]
    patch_acc = acc_sum[valid_patches] / count[valid_patches]
    return patch_conf, patch_acc


class PatchedBinaryCalibrationError(Metric):
    r"""Patch-level calibration error for binary segmentation.

    Instead of computing calibration error per pixel, predictions and targets are aggregated over
    non-overlapping spatial patches of size ``patch_size x patch_size``. For each patch the
    *confidence* is the mean predicted probability and the *accuracy* is the fraction of correctly
    classified pixels. The standard ECE/MCE/RMSCE formula is then applied to these patch-level
    values.

    As input to ``forward`` and ``update`` the metric accepts:

    - ``preds`` (:class:`~torch.Tensor`): Float tensor of shape ``(B, H, W)``
      containing probabilities or logits.  Values outside ``[0, 1]`` are
      treated as logits and sigmoid is applied automatically.
    - ``target`` (:class:`~torch.Tensor`): Int tensor of shape ``(B, H, W)``
      containing binary ground-truth labels ``{0, 1}``.

    Args:
        patch_size: Side length of each square patch in pixels.
        n_bins: Number of confidence bins.  Defaults to ``15``.
        norm: Norm used to aggregate bin-level errors; one of ``'l1'``,
            ``'l2'``, or ``'max'``.  Defaults to ``'l1'``.
        ignore_index: Optional target value to exclude from the metric.
        validate_args: Whether to validate inputs on every update.
        **kwargs: Additional keyword arguments forwarded to
            :class:`torchmetrics.Metric`.
    """

    is_differentiable: bool = False
    higher_is_better: bool = False
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0

    confidences: list[Tensor]
    accuracies: list[Tensor]

    def __init__(
        self,
        patch_size: int,
        n_bins: int = 15,
        norm: Literal["l1", "l2", "max"] = "l1",
        ignore_index: int | None = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if validate_args:
            _patched_binary_calibration_error_arg_validation(patch_size, n_bins, norm, ignore_index)
        self.patch_size = patch_size
        self.n_bins = n_bins
        self.norm = norm
        self.ignore_index = ignore_index
        self.validate_args = validate_args
        self.add_state("confidences", [], dist_reduce_fx="cat")
        self.add_state("accuracies", [], dist_reduce_fx="cat")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update metric states with predictions and targets."""
        if self.validate_args:
            _binary_calibration_error_tensor_validation(preds, target, self.ignore_index)

        if not torch.all((preds >= 0) & (preds <= 1)):
            preds = preds.sigmoid()

        if self.ignore_index is not None:
            valid = (target != self.ignore_index).float()
        else:
            valid = torch.ones_like(preds)

        conf = preds
        acc = ((preds >= 0.5).long() == target).float()

        patch_conf, patch_acc = _pool_patches(conf, acc, valid, self.patch_size)
        self.confidences.append(patch_conf)
        self.accuracies.append(patch_acc)

    def compute(self) -> Tensor:
        """Compute metric."""
        confidences = dim_zero_cat(self.confidences)
        accuracies = dim_zero_cat(self.accuracies)
        return _ce_compute(confidences, accuracies, self.n_bins, norm=self.norm)


class PatchedMulticlassCalibrationError(Metric):
    r"""Patch-level calibration error for multiclass segmentation.

    Instead of computing calibration error per pixel, predictions and targets
    are aggregated over non-overlapping spatial patches of size
    ``patch_size x patch_size``.  For each patch the *confidence* is the mean
    top-1 predicted probability and the *accuracy* is the fraction of correctly
    classified pixels.  The standard ECE/MCE/RMSCE formula is then applied to
    these patch-level values.

    As input to ``forward`` and ``update`` the metric accepts:

    - ``preds`` (:class:`~torch.Tensor`): Float tensor of shape
      ``(B, C, H, W)`` containing class probabilities or logits.  Values
      outside ``[0, 1]`` are treated as logits and softmax is applied
      automatically.
    - ``target`` (:class:`~torch.Tensor`): Int tensor of shape ``(B, H, W)``
      containing ground-truth class indices in ``[0, num_classes - 1]``.

    Args:
        num_classes: Number of classes.
        patch_size: Side length of each square patch in pixels.
        n_bins: Number of confidence bins.  Defaults to ``15``.
        norm: Norm used to aggregate bin-level errors; one of ``'l1'``,
            ``'l2'``, or ``'max'``.  Defaults to ``'l1'``.
        ignore_index: Optional target value to exclude from the metric.
        validate_args: Whether to validate inputs on every update.
        **kwargs: Additional keyword arguments forwarded to
            :class:`torchmetrics.Metric`.
    """

    is_differentiable: bool = False
    higher_is_better: bool = False
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0

    confidences: list[Tensor]
    accuracies: list[Tensor]

    def __init__(
        self,
        num_classes: int,
        patch_size: int,
        num_bins: int = 15,
        norm: Literal["l1", "l2", "max"] = "l1",
        validate_args: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if validate_args:
            _patched_multiclass_calibration_error_arg_validation(
                patch_size, num_classes, num_bins, norm, None
            )
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.num_bins = num_bins
        self.norm = norm
        self.validate_args = validate_args
        self.add_state("confidences", [], dist_reduce_fx="cat")
        self.add_state("accuracies", [], dist_reduce_fx="cat")

    def update(self, preds: Tensor, target: Tensor, ignore_mask: Tensor | None = None) -> None:
        """Update metric states with predictions and targets."""
        if self.validate_args:
            _multiclass_calibration_error_tensor_validation(preds, target, self.num_classes, None)
        if not torch.all((preds >= 0) & (preds <= 1)):
            preds = preds.softmax(dim=1)

        conf, predictions = preds.max(dim=1)  # (B, H, W)
        acc = (predictions == target).float()  # (B, H, W)

        valid = (~ignore_mask).float() if ignore_mask is not None else torch.ones_like(conf)

        patch_conf, patch_acc = _pool_patches(conf, acc, valid, self.patch_size)
        self.confidences.append(patch_conf)
        self.accuracies.append(patch_acc)

    def compute(self) -> Tensor:
        """Compute metric."""
        confidences = dim_zero_cat(self.confidences)
        accuracies = dim_zero_cat(self.accuracies)
        return _ce_compute(confidences, accuracies, self.num_bins, norm=self.norm)


class PatchedCalibrationError:
    def __new__(
        cls: type["PatchedCalibrationError"],
        task: Literal["binary", "multiclass"],
        n_bins: int = 15,
        norm: Literal["l1", "l2", "max"] = "l1",
        num_classes: int | None = None,
        ignore_index: int | None = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> Metric:
        """Initialize task metric."""
        task = ClassificationTaskNoMultilabel.from_str(task)
        kwargs.update(
            {
                "n_bins": n_bins,
                "norm": norm,
                "ignore_index": ignore_index,
                "validate_args": validate_args,
            }
        )
        if task == ClassificationTaskNoMultilabel.BINARY:
            return PatchedBinaryCalibrationError(**kwargs)
        if task == ClassificationTaskNoMultilabel.MULTICLASS:
            if not isinstance(num_classes, int):
                raise ValueError(
                    f"`num_classes` is expected to be `int` but `{type(num_classes)} was passed.`"
                )
            return PatchedMulticlassCalibrationError(num_classes, **kwargs)
        raise ValueError(f"Not handled value: {task}")
