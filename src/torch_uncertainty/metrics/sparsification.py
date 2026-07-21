from typing import cast

import matplotlib.pyplot as plt
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from torch import Tensor
from torchmetrics.metric import Metric
from torchmetrics.utilities.data import dim_zero_cat
from torchmetrics.utilities.plot import _AX_TYPE


class AUSE(Metric):
    is_differentiable = False
    higher_is_better = False
    full_state_update = False
    plot_lower_bound = 0.0
    plot_upper_bound = 100.0
    plot_legend_name = "Sparsification Curves"

    scores: list[Tensor]
    errors: list[Tensor]

    def __init__(self, **kwargs) -> None:
        r"""The Area Under the Sparsification Error curve (AUSE) metric to evaluate
        the quality of the uncertainty estimates, i.e., how much they coincide with
        the true errors.

        Args:
            kwargs: Additional keyword arguments, see `Advanced metric settings <https://torchmetrics.readthedocs.io/en/stable/pages/overview.html#metric-kwargs>`_.

        Inputs:
            - :attr:`scores`: Uncertainty scores of shape :math:`(B,)`. A higher
              score means a higher uncertainty.
            - :attr:`errors`: Errors of shape :math:`(B,)`,

            where :math:`B` is the batch size.

        References:
            [1] `Uncertainty estimates and multi-hypotheses for optical flow. In ECCV, 2018
            <https://arxiv.org/abs/1802.07095>`_.

        Note:
            A higher AUSE means a lower quality of the uncertainty estimates.
        """
        super().__init__(**kwargs)
        self.add_state("scores", default=[], dist_reduce_fx="cat")
        self.add_state("errors", default=[], dist_reduce_fx="cat")

    def update(self, scores: Tensor, errors: Tensor) -> None:  # pyrefly: ignore[bad-override]
        """Store the scores and their associated errors for later computation.

        Args:
            scores: uncertainty scores of shape :math:`(B,)`
            errors: errors of shape :math:`(B,)`
        """
        if scores.ndim != 1 or errors.ndim != 1:
            raise ValueError("Expected `scores` and `errors` to be one-dimensional tensors.")
        if scores.shape != errors.shape:
            raise ValueError(
                "Expected `scores` and `errors` to have the same shape, but got "
                f"{scores.shape} and {errors.shape}."
            )
        if errors.is_floating_point() and not torch.isfinite(errors).all():
            raise ValueError("Expected `errors` to contain only finite values.")
        if torch.any(errors < 0):
            raise ValueError("Expected `errors` to contain only non-negative values.")

        self.scores.append(scores)
        self.errors.append(errors)

    def partial_compute(self) -> tuple[Tensor, Tensor]:
        scores = dim_zero_cat(self.scores)
        errors = dim_zero_cat(self.errors)
        if scores.shape[0] < 2:
            nan = torch.tensor(float("nan"), device=self.device)
            return nan, nan
        error_rates = _ause_rejection_rate_compute(scores, errors)
        optimal_error_rates = _ause_rejection_rate_compute(errors, errors)
        return error_rates, optimal_error_rates

    def compute(self) -> Tensor:
        """Compute the Area Under the Sparsification Error curve (AUSE) based
        on inputs passed to ``update``.

        Returns:
            Tensor: The AUSE.
        """
        error_rates, optimal_error_rates = self.partial_compute()
        if error_rates.ndim == 0:
            return error_rates
        num_samples = error_rates.size(0)
        x = torch.arange(num_samples, device=error_rates.device, dtype=error_rates.dtype)
        x /= num_samples
        y = error_rates - optimal_error_rates
        return torch.trapezoid(y, x)

    def plot(  # pyrefly: ignore[bad-override]
        self,
        ax: _AX_TYPE | None = None,
        plot_oracle: bool = True,
        plot_value: bool = True,
    ) -> tuple[Figure | None, Axes | object]:
        """Plot the sparsification curve corresponding to the inputs passed to
        ``update``, and the oracle sparsification curve.

        Args:
            ax: An matplotlib axis object. If provided will add plot to this axis.
                Defaults to ``None``.
            plot_oracle: Whether to plot the oracle sparsification curve. Defaults to ``True``.
            plot_value: Whether to plot the AUSE value. Defaults to ``True``.

        Returns:
            tuple[[Figure | None], Axes]: Figure object and Axes object
        """
        fig, ax = plt.subplots() if ax is None else (None, ax)
        ax = cast("Axes", ax)

        # Computation of AUSEC
        error_rates, optimal_error_rates = self.partial_compute()
        if error_rates.ndim == 0:
            raise RuntimeError("AUSE requires at least two samples before plotting.")
        num_samples = error_rates.size(0)
        x = torch.arange(num_samples, device=error_rates.device, dtype=error_rates.dtype)
        x /= num_samples
        y = error_rates - optimal_error_rates

        ausec = torch.trapezoid(y, x).item()

        rejection_rates = x.cpu() * 100

        ax.plot(
            rejection_rates,
            error_rates.cpu() * 100,
            label="Model",
        )
        if plot_oracle:
            ax.plot(
                rejection_rates,
                optimal_error_rates.cpu() * 100,
                label="Oracle",
            )

        ax.set_xlabel("Rejection Rate (%)")
        ax.set_ylabel("Error Rate (%)")
        ax.set_xlim(self.plot_lower_bound, self.plot_upper_bound)
        ax.set_ylim(self.plot_lower_bound, self.plot_upper_bound)
        ax.legend(loc="upper right")

        if plot_value:
            ax.text(
                0.02,
                0.02,
                f"AUSEC={ausec:.03}",
                color="black",
                ha="left",
                va="bottom",
                transform=ax.transAxes,
            )

        return fig, ax


def _ause_rejection_rate_compute(
    scores: Tensor,
    errors: Tensor,
) -> Tensor:
    """Compute the cumulative error rates for a given set of scores and errors.

    Args:
        scores: uncertainty scores of shape :math:`(B,)`
        errors: errors of shape :math:`(B,)`
    """
    num_samples = errors.size(0)

    dtype = errors.dtype if errors.is_floating_point() else torch.get_default_dtype()
    ordered_errors = errors[scores.argsort()].to(dtype)

    # At rejection step k, the k samples with the highest uncertainty have
    # been discarded. Since scores are sorted increasingly, the retained
    # samples form the prefix ending at N-k.
    remaining_sum = ordered_errors.cumsum(dim=0).flip(0)
    remaining_count = torch.arange(
        num_samples,
        0,
        -1,
        device=errors.device,
        dtype=dtype,
    )
    error_rates = remaining_sum / remaining_count

    initial_error = ordered_errors.mean()
    if initial_error == 0:
        return torch.zeros_like(error_rates)
    return error_rates / initial_error
