import torch
from torch import Tensor
from torchmetrics import Metric

from torch_uncertainty.metrics.classification.fpr import _fprx_compute

from ._binary import binary_images, binary_target_has_classes


class SegmentationFPR95(Metric):
    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    fpr95: Tensor
    total: Tensor

    def __init__(self, pos_label: int, ignore_index: int | None = None, **kwargs) -> None:
        r"""Image-averaged FPR@95 TPR for dense binary segmentation tasks.

        For each image, a per-pixel False Positive Rate at 95% True Positive Rate is
        computed (see :class:`~torch_uncertainty.metrics.classification.FPR95`) from
        the pixel scores and binary OOD labels. The metric is then averaged over the
        :math:`B` images of the test set:

        .. math::
            \text{FPR95} = \frac{1}{B} \sum_{b=1}^{B} \text{FPR95}_b.

        Image-wise averaging is the convention used in the dense OOD-detection
        literature.

        Images without both positive and negative pixels are excluded because FPR@95
        is undefined. The metric returns ``nan`` if no valid image was observed. A
        one-dimensional input is treated as one image; otherwise, the first dimension
        is the image batch dimension.

        Args:
            pos_label: The positive label in the segmentation OOD detection task
                (typically ``1`` for OOD pixels).
            ignore_index: Optional label value to ignore.
            kwargs: Additional keyword arguments for the underlying
                :class:`~torch_uncertainty.metrics.classification.FPR95` metric.
        """
        super().__init__(**kwargs)
        self.pos_label = pos_label
        self.ignore_index = ignore_index
        self.add_state("fpr95", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:  # pyrefly: ignore[bad-override]
        for image_preds, image_target in binary_images(preds, target):
            if self.ignore_index is not None:
                keep = image_target != self.ignore_index
                image_preds = image_preds[keep]
                image_target = image_target[keep]
            if not binary_target_has_classes(image_target, pos_label=self.pos_label):
                continue
            self.fpr95 += _fprx_compute(
                image_preds,
                image_target,
                recall_level=0.95,
                pos_label=self.pos_label,
            )
            self.total += 1

    def compute(self) -> Tensor:
        if self.total == 0:
            return torch.tensor(torch.nan, device=self.fpr95.device)
        return self.fpr95 / self.total
