import warnings
from typing import Literal

import matplotlib.pyplot as plt
import torch
from torch import Tensor
from torch.distributions import Distribution, Independent
from torchmetrics import Metric
from torchmetrics.utilities.plot import _PLOT_OUT_TYPE


class QuantileCalibrationError(Metric):
    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    covered: Tensor
    total: Tensor
    unsupported: Tensor

    def __init__(
        self,
        num_bins: int = 15,
        norm: Literal["l1", "l2", "max"] = "l1",
        ignore_index: int | None = None,
        validate_args: bool = True,
        **kwargs,
    ) -> None:
        r"""Quantile Calibration Error for regression tasks.

        For each confidence level :math:`\alpha \in (0, 1)`, a well-calibrated
        probabilistic regressor should ensure that a fraction :math:`\alpha` of the
        ground-truth targets lies inside the centered :math:`\alpha`-credible interval
        of the predicted distribution :math:`p_\theta(\cdot \mid x)`. Concretely, let

        .. math::
            \hat{c}(\alpha) = \frac{1}{N} \sum_{i=1}^{N}
            \mathbf{1}\!\left[ y_i \in
            \left[ F^{-1}_{\theta, x_i}\!\left(\tfrac{1 - \alpha}{2}\right),
                   F^{-1}_{\theta, x_i}\!\left(\tfrac{1 + \alpha}{2}\right) \right]
            \right].

        The metric evaluates this coverage on :attr:`num_bins` confidence levels
        :math:`\alpha_k` regularly spaced between ``0.05`` and ``0.95``. It returns

        .. math::
            \operatorname{QCE}_{L_1}
            = \frac{1}{K}\sum_{k=1}^{K}
            \left|\hat{c}(\alpha_k)-\alpha_k\right|,

        with analogous root-mean-square and maximum variants for ``norm="l2"``
        and ``norm="max"``.

        For :class:`~torch.distributions.Independent` distributions, calibration is
        evaluated marginally: every scalar event component contributes one coverage
        observation.

        Args:
            num_bins: Number of confidence levels. Defaults to ``15``.
            norm: Norm used to aggregate the calibration gaps. One of ``"l1"``,
                ``"l2"``, or ``"max"``. Defaults to ``"l1"``.
            ignore_index: Optional target value to ignore. Defaults to ``None``.
            validate_args: Whether to validate input shapes. Defaults to ``True``.
            kwargs: Additional keyword arguments, see `Advanced metric settings
                <https://torchmetrics.readthedocs.io/en/stable/pages/overview.html#metric-kwargs>`_.
        """
        super().__init__(**kwargs)
        if num_bins < 1:
            raise ValueError(f"Expected `num_bins` to be at least 1, but got {num_bins}.")
        if norm not in ("l1", "l2", "max"):
            raise ValueError(f"Expected `norm` to be one of ('l1', 'l2', 'max'), but got {norm}.")

        self.n_bins = num_bins
        self.norm = norm
        self.ignore_index = ignore_index
        self.validate_args = validate_args
        self.register_buffer("conf_intervals", torch.linspace(0.05, 0.95, num_bins))
        self.add_state(
            "covered",
            default=torch.zeros(num_bins, dtype=torch.long),
            dist_reduce_fx="sum",
        )
        self.add_state("total", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")
        self.add_state(
            "unsupported",
            default=torch.tensor(0, dtype=torch.long),
            dist_reduce_fx="sum",
        )

    def update(  # pyrefly: ignore[bad-override]
        self,
        dist: Distribution,
        target: Tensor,
        ignore_mask: Tensor | None = None,
    ) -> None:
        """Update the metric with predictive distributions and targets.

        Args:
            dist: Predicted distribution. It must implement ``icdf``.
            target: Ground-truth values, with one value per predictive distribution.
            ignore_mask: Boolean mask of targets to ignore. A mask over only the batch
                dimensions is expanded over trailing event dimensions. Defaults to ``None``.
        """
        expected_shape = dist.batch_shape + dist.event_shape
        if self.validate_args and target.shape != expected_shape:
            raise ValueError(
                "Expected `target` to have shape equal to the distribution batch and "
                f"event shapes ({expected_shape}), but got {target.shape}."
            )

        marginal_dist = dist.base_dist if isinstance(dist, Independent) else dist
        levels = self.conf_intervals.to(target.device)

        try:
            intervals = [
                (
                    marginal_dist.icdf((1 - level) / 2),
                    marginal_dist.icdf((1 + level) / 2),
                )
                for level in levels
            ]
        except NotImplementedError:
            warnings.warn(
                "The distribution does not support the `icdf()` method. "
                "This metric will therefore return `nan`. Please use a "
                "distribution that implements `icdf()`.",
                UserWarning,
                stacklevel=2,
            )
            self.unsupported += 1
            return

        valid = torch.ones_like(target, dtype=torch.bool)
        if self.ignore_index is not None:
            valid &= target != self.ignore_index
        if ignore_mask is not None:
            ignore_mask = ignore_mask.bool()
            while ignore_mask.ndim < target.ndim:
                ignore_mask = ignore_mask.unsqueeze(-1)
            try:
                ignore_mask = torch.broadcast_to(ignore_mask, target.shape)
            except RuntimeError as err:
                raise ValueError(
                    f"Expected `ignore_mask` to be broadcastable to {target.shape}, "
                    f"but got {ignore_mask.shape}."
                ) from err
            valid &= ~ignore_mask

        inside = torch.stack(
            [(target >= lower) & (target <= upper) & valid for lower, upper in intervals]
        )
        self.covered += inside.reshape(self.n_bins, -1).sum(dim=1)
        self.total += valid.sum()

    def compute(self) -> Tensor:
        """Compute the Quantile Calibration Error."""
        if self.unsupported > 0 or self.total == 0:
            return torch.tensor(torch.nan, device=self.covered.device)

        empirical_coverage = self.covered / self.total
        calibration_gap = (empirical_coverage - self.conf_intervals).abs()
        if self.norm == "l1":
            return calibration_gap.mean()
        if self.norm == "l2":
            return calibration_gap.square().mean().sqrt()
        return calibration_gap.max()

    def plot(self) -> _PLOT_OUT_TYPE:  # pyrefly: ignore[bad-override]
        """Plot empirical coverage against nominal coverage."""
        if self.unsupported > 0:
            raise NotImplementedError(
                "The distribution does not support the `icdf()` method. "
                "Please use a distribution that implements `icdf()`."
            )
        if self.total == 0:
            raise RuntimeError("QuantileCalibrationError requires at least one valid target.")

        nominal = self.conf_intervals.detach().cpu() * 100
        empirical = (self.covered / self.total).detach().cpu() * 100
        fig, ax = plt.subplots()
        ax.plot(nominal, empirical, marker="o", label="Model")
        ax.plot([0, 100], [0, 100], linestyle="--", color="black", label="Ideal")
        ax.set_xlabel("Nominal Coverage (%)")
        ax.set_ylabel("Empirical Coverage (%)")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.legend()
        return fig, ax
