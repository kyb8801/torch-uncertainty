import matplotlib.pyplot as plt
import pytest
import torch
from torch.distributions import Distribution, Exponential, Independent, Normal

from torch_uncertainty.metrics import QuantileCalibrationError


class TestQuantileCalibrationError:
    """Testing the QuantileCalibrationError metric class."""

    def test_calibrated_normal(self) -> None:
        torch.manual_seed(42)
        dist = Normal(torch.zeros(10_000), torch.ones(10_000))
        qce = QuantileCalibrationError()

        qce.update(dist, dist.sample())

        assert qce.compute().item() < 0.02

    def test_calibrated_asymmetric_distribution(self) -> None:
        num_samples = 10_000
        quantiles = (torch.arange(num_samples) + 0.5) / num_samples
        dist = Exponential(torch.ones(num_samples))
        targets = dist.icdf(quantiles)
        qce = QuantileCalibrationError()

        qce.update(dist, targets)

        # This catches the previous density-threshold implementation, which
        # returned approximately 0.4 for a calibrated exponential distribution.
        assert qce.compute().item() < 2e-4

    def test_independent_distribution_uses_marginal_coverage(self) -> None:
        num_samples = 1_000
        quantiles = ((torch.arange(num_samples) + 0.5) / num_samples).unsqueeze(1)
        quantiles = quantiles.expand(-1, 2)
        base_dist = Normal(torch.zeros_like(quantiles), torch.ones_like(quantiles))
        dist = Independent(base_dist, 1)
        qce = QuantileCalibrationError()

        qce.update(dist, base_dist.icdf(quantiles))

        assert qce.compute().item() < 2e-3

    def test_ignore_mask_over_batch_dimensions(self) -> None:
        num_samples = 1_000
        quantiles = ((torch.arange(num_samples) + 0.5) / num_samples).unsqueeze(1)
        dist = Normal(torch.zeros_like(quantiles), torch.ones_like(quantiles))
        ignore_mask = torch.zeros(num_samples, dtype=torch.bool)
        ignore_mask[:100] = True
        qce = QuantileCalibrationError()

        qce.update(dist, dist.icdf(quantiles), ignore_mask=ignore_mask)

        assert qce.total == 900
        assert torch.isfinite(qce.compute())

    @pytest.mark.parametrize("norm", ["l1", "l2", "max"])
    def test_norms(self, norm: str) -> None:
        dist = Normal(torch.zeros(4), torch.ones(4))
        qce = QuantileCalibrationError(num_bins=3, norm=norm)
        qce.update(dist, torch.tensor([-2.0, -0.5, 0.5, 2.0]))

        gap = (qce.covered / qce.total - qce.conf_intervals).abs()
        expected = {
            "l1": gap.mean(),
            "l2": gap.square().mean().sqrt(),
            "max": gap.max(),
        }[norm]
        torch.testing.assert_close(qce.compute(), expected)

    def test_unsupported_distribution_state_is_reset(self) -> None:
        qce = QuantileCalibrationError()
        unsupported_dist = Distribution(batch_shape=torch.Size([2]), validate_args=False)
        with pytest.warns(UserWarning, match="does not support"):
            qce.update(unsupported_dist, torch.zeros(2))
        assert torch.isnan(qce.compute())

        qce.reset()
        dist = Normal(torch.zeros(2), torch.ones(2))
        qce.update(dist, torch.zeros(2))
        assert torch.isfinite(qce.compute())

    def test_plot(self) -> None:
        dist = Normal(torch.zeros(100), torch.ones(100))
        qce = QuantileCalibrationError()
        qce.update(dist, dist.sample())

        fig, ax = qce.plot()

        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        assert ax.get_xlabel() == "Nominal Coverage (%)"
        assert ax.get_ylabel() == "Empirical Coverage (%)"
        plt.close(fig)

    def test_invalid_arguments_and_shapes(self) -> None:
        with pytest.raises(ValueError, match="num_bins"):
            QuantileCalibrationError(num_bins=0)
        with pytest.raises(ValueError, match="norm"):
            QuantileCalibrationError(norm="invalid")

        qce = QuantileCalibrationError()
        with pytest.raises(ValueError, match="target"):
            qce.update(Normal(torch.zeros(2), torch.ones(2)), torch.zeros(3))

    def test_ignore_index(self) -> None:
        dist = Normal(torch.zeros(3), torch.ones(3))
        qce = QuantileCalibrationError(ignore_index=-1)
        qce.update(dist, torch.tensor([0.0, -1.0, 0.0]))

        assert qce.total == 2
        assert torch.isfinite(qce.compute())

    def test_shape_validation_can_be_disabled(self) -> None:
        dist = Normal(torch.zeros(2), torch.ones(2))
        qce = QuantileCalibrationError(validate_args=False)
        qce.update(dist, torch.zeros(2))
        assert torch.isfinite(qce.compute())

    def test_invalid_ignore_mask_shape(self) -> None:
        dist = Normal(torch.zeros(2, 3), torch.ones(2, 3))
        qce = QuantileCalibrationError()

        with pytest.raises(ValueError, match="broadcastable"):
            qce.update(dist, torch.zeros(2, 3), ignore_mask=torch.zeros(4))

    def test_plot_requires_supported_distribution_and_valid_targets(self) -> None:
        empty = QuantileCalibrationError()
        with pytest.raises(RuntimeError, match="at least one valid target"):
            empty.plot()

        unsupported = QuantileCalibrationError()
        dist = Distribution(batch_shape=torch.Size([2]), validate_args=False)
        with pytest.warns(UserWarning, match="does not support"):
            unsupported.update(dist, torch.zeros(2))
        with pytest.raises(NotImplementedError, match="does not support"):
            unsupported.plot()
