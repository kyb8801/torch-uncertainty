import math
from typing import Literal

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from torch_uncertainty.post_processing.abstract import PostProcessing
from torch_uncertainty.utils.checks import check_interval_shapes


class ConformalRegCQR(PostProcessing):
    q_hat: Tensor | None = None

    def __init__(
        self,
        alpha: float,
        model: nn.Module | None = None,
        device: Literal["cpu", "cuda"] | torch.device | None = None,
    ) -> None:
        r"""Conformalized Quantile Regression (CQR; Romano et al., 2019).

        Post-hoc calibration of a quantile-regression model that turns its two
        predicted quantiles into a prediction interval with the *marginal coverage
        guarantee*

        .. math::
            \mathbb{P}\!\left[ Y \in [\,\hat{l}(X),\, \hat{u}(X)\,] \right] \geq 1 - \alpha,

        provided the calibration and test points are exchangeable. The wrapped
        ``model`` is expected to output the lower and upper quantiles
        :math:`\hat{q}_{\alpha/2}` and :math:`\hat{q}_{1-\alpha/2}` as a tensor of
        shape ``(batch, 2)``. At fit time, the two-sided conformity scores

        .. math::
            E_i = \max\{\,\hat{q}_{\alpha/2}(x_i) - y_i,\; y_i - \hat{q}_{1-\alpha/2}(x_i)\,\}

        are computed on a held-out calibration set and their finite-sample
        :math:`(1-\alpha)`-quantile :math:`\hat{q}` is stored in :attr:`q_hat`. At
        test time the calibrated interval is

        .. math::
            \mathcal{C}(x) = [\,\hat{q}_{\alpha/2}(x) - \hat{q},\; \hat{q}_{1-\alpha/2}(x) + \hat{q}\,].

        Args:
            alpha: Target mis-coverage level :math:`\alpha \in (0, 1)`. A smaller
                :math:`\alpha` yields wider intervals.
            model: Quantile-regression model returning the lower and upper quantiles
                as a ``(batch, 2)`` tensor. Can be set later via :meth:`set_model`.
                Defaults to ``None``.
            device: Device to run the post-processing on. Defaults to ``None``.

        Reference:
            - `Romano, Y., Patterson, E., & Candès, E. (2019). Conformalized Quantile
              Regression <https://arxiv.org/abs/1905.03222>`_.
        """
        super().__init__(model=model)
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1). Got {alpha}.")
        self.alpha = alpha
        self.device = device or "cpu"

    @torch.no_grad()
    def fit(self, dataloader: DataLoader) -> None:
        r"""Calibrate the conformal correction on a held-out calibration set.

        Runs the quantile model over the calibration dataloader, computes the
        two-sided conformity scores, and stores their finite-sample
        :math:`(1-\alpha)`-quantile in :attr:`q_hat`.
        """
        assert self.model is not None
        self.model.eval()

        lowers, uppers, targets = [], [], []
        for inputs, target in dataloader:
            quantiles = self.model(inputs.to(self.device))
            target = target.to(self.device).reshape(-1)
            check_interval_shapes(quantiles[:, 0], quantiles[:, 1], target)
            lowers.append(quantiles[:, 0])
            uppers.append(quantiles[:, 1])
            targets.append(target)

        lower = torch.cat(lowers)
        upper = torch.cat(uppers)
        target = torch.cat(targets)

        scores = torch.maximum(lower - target, target - upper)
        n = scores.numel()
        level = min(math.ceil((n + 1) * (1.0 - self.alpha)) / n, 1.0)
        self.q_hat = torch.quantile(scores, level, interpolation="higher")
        self.trained = True

    @torch.no_grad()
    def forward(self, inputs: Tensor) -> Tensor:
        """Return the calibrated ``(batch, 2)`` ``[lower, upper]`` interval."""
        if self.model is None:  # coverage: ignore
            raise RuntimeError("Model must be set before calling forward().")
        quantiles = self.model(inputs.to(self.device))
        lower = quantiles[:, 0] - self.quantile
        upper = quantiles[:, 1] + self.quantile
        return torch.stack((lower, upper), dim=-1)

    @property
    def quantile(self) -> Tensor:
        if self.q_hat is None:
            raise RuntimeError("Quantile q_hat is not set. Run `.fit()` first.")
        return self.q_hat
