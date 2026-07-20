from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.utilities.data import dim_zero_cat


class BrierScore(Metric):
    is_differentiable = True
    higher_is_better = False
    full_state_update = False

    values: list[Tensor]
    total: Tensor

    def __init__(
        self,
        num_classes: int,
        top_class: bool = False,
        reduction: Literal["mean", "sum", "none"] | None = "mean",
        **kwargs,
    ) -> None:
        r"""Compute the Brier score.

        The Brier Score measures the mean squared difference between predicted
        probabilities and actual target values. It is used to evaluate the
        accuracy of probabilistic predictions, where a lower score indicates
        better calibration and prediction quality.

        Given predicted probabilities :math:`\hat{p}_{i,c}` and one-hot encoded
        targets :math:`y_{i,c}` for :math:`N` samples and :math:`C` classes:

        .. math::

            \text{BS} = \frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C}
            \left( \hat{p}_{i,c} - y_{i,c} \right)^2

        When ``top_class=True``, only the top predicted class is considered:

        .. math::

            \text{BS}_{\text{top}} = \frac{1}{N} \sum_{i=1}^{N}
            \left( \hat{p}_i - a_i \right)^2

        where :math:`\hat{p}_i = \max_c \hat{p}_{i,c}` is the highest predicted
        probability and :math:`a_i = \mathbf{1}[\hat{y}_i = y_i]` indicates
        whether the top prediction is correct.

        Args:
            num_classes: Number of classes.
            top_class: If True, computes the Brier score for the top predicted class only.
                Defaults to ``False``.
            reduction: Determines how to reduce the score across the
                batch dimension:

                - ``'mean'`` [default]: Averages the score across samples.
                - ``'sum'``: Sums the score across samples.
                - ``'none'`` or ``None``: Returns the score for each sample.

            kwargs: Additional keyword arguments, see `Advanced metric settings
                <https://torchmetrics.readthedocs.io/en/stable/pages/overview.html#metric-kwargs>`_.

        Inputs:
            - :attr:`probs`: :math:`(B, C)` or :math:`(B, N, C)` for multiclass
                predictions. For binary predictions (``num_classes=1``), :math:`(B)`,
                :math:`(B, 1)`, or :math:`(B, N, 1)`.
            - :attr:`target`: :math:`(B)` or :math:`(B, C)`
                Ground truth class labels or one-hot encoded targets.

            where:
                :math:`B` is the batch size,
                :math:`C` is the number of classes,
                :math:`N` is the number of estimators.

        Note:
            If :attr:`probs` is a 3D tensor, the metric computes the mean of
            the Brier score over the estimators, as:
            :math:`t = \frac{1}{N} \sum_{i=0}^{N-1} BrierScore(probs[:,i,:], target)`.

        Warning:
            Ensure that the probabilities in :attr:`probs` are normalized to sum
            to one before passing them to the metric.

        Raises:
            ValueError: If :attr:`reduction` is not one of ``'mean'``, ``'sum'``,
                ``'none'`` or ``None``.

        Examples:
            >>> from torch_uncertainty.metrics.classification.brier_score import BrierScore
            # Example 1: Binary Classification
            >>> probs = torch.tensor([[0.8, 0.2], [0.3, 0.7]])
            >>> target = torch.tensor([0, 1])
            >>> metric = BrierScore(num_classes=2)
            >>> metric.update(probs, target)
            >>> score = metric.compute()
            >>> print(score)
            tensor(0.1299)
            # Example 2: Multi-Class Classification
            >>> probs = torch.tensor([[0.6, 0.3, 0.1], [0.2, 0.5, 0.3]])
            >>> target = torch.tensor([0, 2])
            >>> metric = BrierScore(num_classes=3, reduction="mean")
            >>> metric.update(probs, target)
            >>> score = metric.compute()
            >>> print(score)
            tensor(0.5199)

        References:
            [1] `Wikipedia entry for the Brier score
            <https://en.wikipedia.org/wiki/Brier_score>`_.
        """
        super().__init__(**kwargs)

        allowed_reduction = ("sum", "mean", "none", None)
        if reduction not in allowed_reduction:
            raise ValueError(
                "Expected argument `reduction` to be one of ",
                f"{allowed_reduction} but got {reduction}",
            )

        self.num_classes = num_classes
        self.top_class = top_class
        self.reduction = reduction

        if self.reduction in ["mean", "sum"]:
            self.add_state(
                "values",
                default=torch.tensor(0.0),
                dist_reduce_fx="sum",
            )
        else:
            self.add_state(
                "values",
                default=[],
                dist_reduce_fx="cat",
            )
        self.add_state(
            "total",
            default=torch.tensor(0),
            dist_reduce_fx="sum",
        )

    def update(
        self,
        probs: Tensor,
        target: Tensor,
    ) -> None:  # pyrefly: ignore[bad-override]
        """Update the current Brier score with a new tensor of probabilities.

        Args:
            probs: A probability tensor of shape
                (batch, num_estimators, num_classes) or
                (batch, num_classes)
            target: A tensor of ground truth labels of shape
                (batch, num_classes) or (batch)
        """
        probs, target = self._format_inputs(probs, target)

        if self.top_class:
            if self.num_classes == 1:
                probs = torch.cat((1 - probs, probs), dim=-1)
                target = torch.cat((1 - target, target), dim=-1)

            confidence, indices = probs.max(dim=-1)
            correct = target.expand(-1, probs.size(1), -1).gather(
                dim=-1,
                index=indices.unsqueeze(-1),
            )
            brier_score = (confidence - correct.squeeze(-1)).square()
        else:
            brier_score = (probs - target).square().sum(dim=-1)

        # Average over estimators immediately so every sample has the same
        # weight, independently of the number of estimators in each update.
        brier_score = brier_score.mean(dim=1)
        batch_size = brier_score.size(0)

        if self.reduction is None or self.reduction == "none":
            self.values.append(brier_score)
        else:
            self.values += brier_score.sum()
            self.total += batch_size

    def _format_inputs(
        self,
        probs: Tensor,
        target: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Normalize inputs to (batch, estimators, classes)."""
        if probs.ndim == 1:
            if self.num_classes != 1:
                raise ValueError("One-dimensional `probs` are only supported for binary tasks.")
            probs = probs[:, None, None]
        elif probs.ndim == 2:
            probs = probs.unsqueeze(1)
        elif probs.ndim != 3:
            raise ValueError(
                "Expected `probs` to have shape (batch, num_classes) or "
                f"(batch, num_estimators, num_classes), but got {probs.shape}."
            )

        if probs.shape[-1] != self.num_classes:
            raise ValueError(f"Expected {self.num_classes} classes, but got {probs.shape[-1]}.")

        if target.ndim == 2 and target.shape[-1] == 1:
            target = target.squeeze(-1)

        if target.ndim == 1:
            target = (
                target.unsqueeze(-1)
                if self.num_classes == 1
                else F.one_hot(target, self.num_classes)
            )
        elif target.ndim != 2 or target.shape[-1] != self.num_classes:
            raise ValueError(
                "Expected `target` to have shape (batch) or "
                f"(batch, num_classes), but got {target.shape}."
            )

        if probs.shape[0] != target.shape[0]:
            raise ValueError("Expected `probs` and `target` to have the same batch size.")

        target = target.to(
            device=probs.device,
            dtype=probs.dtype,
        ).unsqueeze(1)

        return probs, target

    def compute(self) -> Tensor:
        """Compute the final Brier score based on inputs passed to ``update``.

        Returns:
            Tensor: The final value(s) for the Brier score.
        """
        values = dim_zero_cat(self.values)
        if self.reduction == "sum":
            return values.sum()
        if self.reduction == "mean":
            return values.sum() / self.total
        return values
