from typing import Any

import torch
from torch import Tensor
from torchmetrics import Metric

from torch_uncertainty.utils import check_interval_shapes


class IntervalCoverage(Metric):
    is_differentiable: bool = False
    higher_is_better: bool | None = None
    full_state_update: bool = False
    covered: Tensor
    total: Tensor

    def __init__(self, **kwargs: Any) -> None:
        r"""Prediction Interval Coverage Probability (PICP).

        Given predicted lower and upper bounds :math:`\hat{l}_i` and :math:`\hat{u}_i` of a
        central prediction interval, PICP is the empirical fraction of targets that fall
        inside the interval:

        .. math::
            \text{PICP} = \frac{1}{N} \sum_{i=1}^{N}
            \mathbf{1}\!\left[ \hat{l}_i \le y_i \le \hat{u}_i \right].

        A well-calibrated interval built for a nominal coverage :math:`1 - \alpha` should
        yield :math:`\text{PICP} \approx 1 - \alpha`. Coverage is **gameable on its own** —
        an arbitrarily wide interval reaches :math:`\text{PICP} = 1` — so it should always be
        reported together with an interval-width metric such as
        :class:`~torch_uncertainty.metrics.regression.MeanIntervalWidth` or a proper interval
        score such as :class:`~torch_uncertainty.metrics.regression.IntervalScore`.

        Args:
            kwargs: Additional keyword arguments, see `Advanced metric settings
                <https://torchmetrics.readthedocs.io/en/stable/pages/overview.html#metric-kwargs>`_.

        Note:
            Inputs of any shape are accepted and flattened, so the metric returns the coverage
            rate over all elements (e.g. batch and output dimensions pooled together).

        Example:

        .. code-block:: python

            import torch
            from torch_uncertainty.metrics.regression import IntervalCoverage

            lower = torch.tensor([0.0, 1.0, 2.0, 3.0])
            upper = torch.tensor([2.0, 3.0, 4.0, 5.0])
            target = torch.tensor([1.0, 5.0, 3.0, 4.0])  # 3 of 4 inside

            metric = IntervalCoverage()
            metric.update(lower, upper, target)
            print(metric.compute())
            # tensor(0.7500)
        """
        super().__init__(**kwargs)
        self.add_state("covered", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, lower: Tensor, upper: Tensor, target: Tensor) -> None:
        """Update state with a batch of interval bounds and targets.

        Args:
            lower: The predicted lower bounds of the interval.
            upper: The predicted upper bounds of the interval.
            target: The ground-truth targets.
        """
        check_interval_shapes(lower, upper, target)
        inside = (target >= lower) & (target <= upper)
        self.covered += inside.sum()
        self.total += target.numel()

    def compute(self) -> Tensor:
        """Compute the prediction interval coverage probability."""
        return self.covered.float() / self.total


class MeanIntervalWidth(Metric):
    is_differentiable: bool = False
    higher_is_better: bool = False
    full_state_update: bool = False
    width_sum: Tensor
    total: Tensor

    def __init__(self, **kwargs: Any) -> None:
        r"""Mean Prediction Interval Width (MPIW), a.k.a. sharpness.

        The mean width of the predicted central prediction intervals:

        .. math::
            \text{MPIW} = \frac{1}{N} \sum_{i=1}^{N} \left( \hat{u}_i - \hat{l}_i \right).

        MPIW measures the sharpness of the intervals. It is only meaningful **jointly** with a
        coverage metric such as :class:`~torch_uncertainty.metrics.regression.IntervalCoverage`:
        narrower is better *at equal coverage*, since width can be reduced trivially by
        sacrificing coverage.

        Args:
            kwargs: Additional keyword arguments, see `Advanced metric settings
                <https://torchmetrics.readthedocs.io/en/stable/pages/overview.html#metric-kwargs>`_.

        Note:
            Inputs of any shape are accepted and flattened, so the metric returns the mean width
            over all elements.

        Example:

        .. code-block:: python

            import torch
            from torch_uncertainty.metrics.regression import MeanIntervalWidth

            lower = torch.tensor([0.0, 1.0, 2.0])
            upper = torch.tensor([2.0, 3.0, 5.0])  # widths 2, 2, 3

            metric = MeanIntervalWidth()
            metric.update(lower, upper)
            print(metric.compute())
            # tensor(2.3333)
        """
        super().__init__(**kwargs)
        self.add_state("width_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, lower: Tensor, upper: Tensor) -> None:
        """Update state with a batch of interval bounds.

        Args:
            lower: The predicted lower bounds of the interval.
            upper: The predicted upper bounds of the interval.
        """
        check_interval_shapes(lower, upper)
        self.width_sum += (upper - lower).sum()
        self.total += lower.numel()

    def compute(self) -> Tensor:
        """Compute the mean interval width."""
        return self.width_sum / self.total


class IntervalScore(Metric):
    is_differentiable: bool = False
    higher_is_better: bool = False
    full_state_update: bool = False
    score_sum: Tensor
    total: Tensor

    def __init__(self, coverage: float, **kwargs: Any) -> None:
        r"""Interval (Winkler) score for a central prediction interval.

        A proper scoring rule that rewards sharp intervals while penalizing targets that fall
        outside them. For a central prediction interval :math:`[\hat{l}_i, \hat{u}_i]` built for
        a nominal coverage :math:`1 - \alpha` (so that :math:`\hat{l}_i` and :math:`\hat{u}_i`
        are the :math:`\alpha/2` and :math:`1 - \alpha/2` predictive quantiles), the score is

        .. math::
            S_\alpha(\hat{l}_i, \hat{u}_i; y_i) = (\hat{u}_i - \hat{l}_i)
            + \frac{2}{\alpha} (\hat{l}_i - y_i)\, \mathbf{1}\!\left[y_i < \hat{l}_i\right]
            + \frac{2}{\alpha} (y_i - \hat{u}_i)\, \mathbf{1}\!\left[y_i > \hat{u}_i\right],

        and the metric returns its mean over all elements. Lower is better. Unlike coverage and
        width taken separately, the interval score penalizes width and miscoverage jointly, so a
        trivially wide interval no longer scores well.

        Args:
            coverage: The nominal coverage :math:`1 - \alpha` of the interval, in :math:`(0, 1)`
                (e.g. ``0.9`` for a 90% interval). Sets the miscoverage penalty factor
                :math:`2 / \alpha`.
            kwargs: Additional keyword arguments, see `Advanced metric settings
                <https://torchmetrics.readthedocs.io/en/stable/pages/overview.html#metric-kwargs>`_.

        Note:
            Inputs of any shape are accepted and flattened, so the metric returns the mean score
            over all elements.

        Reference:
            [1] `Gneiting & Raftery, Strictly Proper Scoring Rules, Prediction, and Estimation,
            JASA 2007 <https://doi.org/10.1198/016214506000001437>`_ (Eq. 43).

        Example:

        .. code-block:: python

            import torch
            from torch_uncertainty.metrics.regression import IntervalScore

            lower = torch.tensor([0.0, 1.0])
            upper = torch.tensor([2.0, 3.0])
            target = torch.tensor([1.0, 5.0])  # 2nd target misses the upper bound by 2

            metric = IntervalScore(coverage=0.9)
            metric.update(lower, upper, target)
            print(metric.compute())
            # width mean = 2; penalty on 2nd = (2/0.1)*2 = 40; mean = (2 + 42)/2 = 22
            # tensor(22.)
        """
        super().__init__(**kwargs)
        if not 0.0 < coverage < 1.0:
            raise ValueError(f"coverage must be in the open interval (0, 1), got {coverage}.")
        self.coverage = coverage
        self.alpha = 1.0 - coverage
        self.add_state("score_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, lower: Tensor, upper: Tensor, target: Tensor) -> None:
        """Update state with a batch of interval bounds and targets.

        Args:
            lower: The predicted lower bounds of the interval.
            upper: The predicted upper bounds of the interval.
            target: The ground-truth targets.
        """
        check_interval_shapes(lower, upper, target)
        width = upper - lower
        below = (lower - target).clamp(min=0)
        above = (target - upper).clamp(min=0)
        score = width + (2.0 / self.alpha) * (below + above)
        self.score_sum += score.sum()
        self.total += target.numel()

    def compute(self) -> Tensor:
        """Compute the mean interval (Winkler) score."""
        return self.score_sum / self.total
