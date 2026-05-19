from torch import Tensor, distributions
from torchmetrics.utilities.data import dim_zero_cat

from torch_uncertainty.metrics import CategoricalNLL


class DistributionNLL(CategoricalNLL):
    def update(  # pyrefly: ignore[bad-override]
        self,
        dist: distributions.Distribution,
        target: Tensor,
        padding_mask: Tensor | None = None,
    ) -> None:
        """Update state with the predicted distributions and the targets.

        Args:
            dist: Predicted distributions.
            target: Ground truth labels.
            padding_mask: Optional padding mask. Sets the loss to 0 for padded values. Defaults to
                ``None``.
        """
        nlog_prob = -dist.log_prob(target)
        if padding_mask is not None:
            nlog_prob = nlog_prob.masked_fill(padding_mask, float("nan"))
        if self.reduction is None or self.reduction == "none":
            self.values.append(nlog_prob)
        else:
            self.values += nlog_prob.nansum()
            self.total += padding_mask.sum() if padding_mask is not None else target.numel()

    def compute(self) -> Tensor:
        """Compute NLL based on inputs passed to ``update``."""
        values = dim_zero_cat(self.values)

        if self.reduction == "sum":
            return values.nansum()
        if self.reduction == "mean":
            return values.nansum() / self.total
        # reduction is None or "none"
        return values
