from typing import Any

import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.functional.classification import binary_average_precision

from ._binary import binary_images, binary_target_has_classes


class SegmentationBinaryAveragePrecision(Metric):
    is_differentiable = False
    higher_is_better = True
    full_state_update = False

    binary_aupr: Tensor
    total: Tensor

    def __init__(
        self,
        thresholds: int | list[float] | Tensor | None = None,
        ignore_index: int | None = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> None:
        r"""Image-averaged binary Average Precision for dense segmentation tasks.

        Per-image Average Precision summarises the precision-recall curve obtained by
        sweeping a threshold over the pixel scores of image :math:`b`:

        .. math::
            \text{AP}_b = \sum_{k} \left( R_b(k) - R_b(k-1) \right) P_b(k),

        where :math:`P_b(k)` and :math:`R_b(k)` are the precision and recall at the
        :math:`k`-th threshold. The final metric is averaged over all :math:`B` images:

        .. math::
            \text{AP} = \frac{1}{B} \sum_{b=1}^{B} \text{AP}_b.

        As for :class:`SegmentationBinaryAUROC`, image-wise averaging is the convention
        used in the dense OOD-detection literature.

        Images without positive pixels are excluded because their Average Precision is
        undefined. The metric returns ``nan`` if no valid image was observed. A
        one-dimensional input is treated as one image; otherwise, the first dimension
        is the image batch dimension.

        Args:
            thresholds: Optional explicit thresholds for the PR curve, see
                :class:`~torchmetrics.classification.BinaryAveragePrecision`.
            ignore_index: Optional label value to ignore.
            validate_args: Whether to validate input arguments.
            kwargs: Additional keyword arguments, see `Advanced metric settings
                <https://torchmetrics.readthedocs.io/en/stable/pages/overview.html#metric-kwargs>`_.
        """
        super().__init__(**kwargs)
        self.thresholds = thresholds
        self.ignore_index = ignore_index
        self.validate_args = validate_args
        self.add_state("binary_aupr", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:  # pyrefly: ignore[bad-override]
        for image_preds, image_target in binary_images(preds, target):
            if not binary_target_has_classes(
                image_target,
                ignore_index=self.ignore_index,
                require_negative=False,
            ):
                continue
            thresholds = self.thresholds
            if isinstance(thresholds, Tensor):
                thresholds = thresholds.to(image_preds.device)
            self.binary_aupr += binary_average_precision(
                image_preds,
                image_target,
                thresholds=thresholds,
                ignore_index=self.ignore_index,
                validate_args=self.validate_args,
            )
            self.total += 1

    def compute(self) -> Tensor:
        if self.total == 0:
            return torch.tensor(torch.nan, device=self.binary_aupr.device)
        return self.binary_aupr / self.total
