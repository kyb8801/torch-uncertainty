from typing import Any

import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.functional.classification import binary_auroc

from ._binary import binary_images, binary_target_has_classes


class SegmentationBinaryAUROC(Metric):
    is_differentiable = False
    higher_is_better = True
    full_state_update = False

    binary_auroc: Tensor
    total: Tensor

    def __init__(
        self,
        max_fpr: float | None = None,
        thresholds: int | list[float] | Tensor | None = None,
        ignore_index: int | None = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> None:
        r"""Image-averaged binary AUROC for dense binary segmentation tasks.

        At each image, a per-pixel binary AUROC is computed from the pixel scores
        :math:`s_{ij}` and binary labels :math:`y_{ij} \in \{0, 1\}`:

        .. math::
            \text{AUROC}_b = \int_0^1 \text{TPR}_b\!\left(\text{FPR}_b^{-1}(u)\right) \mathrm{d}u,

        where TPR and FPR are computed by sweeping a threshold over the pixel-level
        scores of image :math:`b`. The final metric is the average over all images:

        .. math::
            \text{AUROC} = \frac{1}{B} \sum_{b=1}^{B} \text{AUROC}_b.

        This image-wise averaging is the convention used in the dense OOD-detection
        literature (e.g., MUAD) and behaves better than computing AUROC over the
        flattened set of all pixels when image sizes or OOD prevalences vary.

        Images without both positive and negative pixels are excluded because their
        AUROC is undefined. The metric returns ``nan`` if no valid image was observed.
        A one-dimensional input is treated as one image; otherwise, the first dimension
        is the image batch dimension.

        Args:
            max_fpr: If set, computes the partial AUROC up to this FPR value
                (passed to :class:`~torchmetrics.classification.BinaryAUROC`).
            thresholds: Optional explicit thresholds to use when computing the ROC.
            ignore_index: Optional label value to ignore.
            validate_args: Whether to validate input arguments.
            kwargs: Additional keyword arguments, see `Advanced metric settings
                <https://torchmetrics.readthedocs.io/en/stable/pages/overview.html#metric-kwargs>`_.
        """
        super().__init__(**kwargs)
        self.max_fpr = max_fpr
        self.thresholds = thresholds
        self.ignore_index = ignore_index
        self.validate_args = validate_args
        self.add_state("binary_auroc", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:  # pyrefly: ignore[bad-override]
        for image_preds, image_target in binary_images(preds, target):
            if not binary_target_has_classes(image_target, ignore_index=self.ignore_index):
                continue
            thresholds = self.thresholds
            if isinstance(thresholds, Tensor):
                thresholds = thresholds.to(image_preds.device)
            self.binary_auroc += binary_auroc(
                image_preds,
                image_target,
                max_fpr=self.max_fpr,
                thresholds=thresholds,
                ignore_index=self.ignore_index,
                validate_args=self.validate_args,
            )
            self.total += 1

    def compute(self) -> Tensor:
        if self.total == 0:
            return torch.tensor(torch.nan, device=self.binary_auroc.device)
        return self.binary_auroc / self.total
