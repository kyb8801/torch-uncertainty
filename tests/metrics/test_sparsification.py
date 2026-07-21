import matplotlib.pyplot as plt
import pytest
import torch

from torch_uncertainty.metrics import AUSE


class TestAUSE:
    """Testing the AUSE metric class."""

    def test_compute_zero(self) -> None:
        values = torch.as_tensor([0.1, 0.2, 0.3, 0.4, 0.5])
        metric = AUSE()
        metric.update(values, values)
        assert metric.compute() == 0

    def test_known_reverse_ranking(self) -> None:
        errors = torch.tensor([1.0, 2.0, 3.0, 4.0])
        scores = -errors
        metric = AUSE()

        metric.update(scores, errors)

        model_curve, oracle_curve = metric.partial_compute()
        torch.testing.assert_close(model_curve, torch.tensor([1.0, 1.2, 1.4, 1.6]))
        torch.testing.assert_close(oracle_curve, torch.tensor([1.0, 0.8, 0.6, 0.4]))
        torch.testing.assert_close(metric.compute(), torch.tensor(0.45))

    def test_zero_errors_have_zero_ause(self) -> None:
        metric = AUSE()
        metric.update(torch.tensor([0.3, 0.1, 0.2]), torch.zeros(3))
        torch.testing.assert_close(metric.compute(), torch.tensor(0.0))

    def test_preserves_dtype(self) -> None:
        metric = AUSE()
        scores = torch.tensor([0.3, 0.1, 0.2], dtype=torch.float64)
        errors = torch.tensor([3.0, 1.0, 2.0], dtype=torch.float64)
        metric.update(scores, errors)
        assert metric.compute().dtype == torch.float64

    def test_plot(self) -> None:
        scores = torch.as_tensor([0.2, 0.1, 0.5, 0.3, 0.4])
        values = torch.as_tensor([0.1, 0.2, 0.3, 0.4, 0.5])
        metric = AUSE()
        metric.update(scores, values)
        fig, ax = metric.plot()
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        assert ax.get_xlabel() == "Rejection Rate (%)"
        assert ax.get_ylabel() == "Error Rate (%)"
        plt.close(fig)

        external_fig, external_ax = plt.subplots()
        returned_fig, returned_ax = metric.plot(ax=external_ax)
        assert returned_fig is None
        assert returned_ax is external_ax
        plt.close(external_fig)

        metric = AUSE()
        metric.update(scores, values)
        fig, ax = metric.plot(plot_oracle=False, plot_value=False)
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        assert ax.get_xlabel() == "Rejection Rate (%)"
        assert ax.get_ylabel() == "Error Rate (%)"
        plt.close(fig)

    def test_compute_nan(self) -> None:
        probs = torch.tensor([0.9])
        targets = torch.tensor([1.0])
        metric = AUSE()
        assert torch.isnan(metric(probs, targets))

    def test_invalid_inputs(self) -> None:
        metric = AUSE()
        with pytest.raises(ValueError, match="one-dimensional"):
            metric.update(torch.ones(1, 2), torch.ones(1, 2))
        with pytest.raises(ValueError, match="same shape"):
            metric.update(torch.ones(2), torch.ones(3))
        with pytest.raises(ValueError, match="non-negative"):
            metric.update(torch.ones(2), torch.tensor([1.0, -1.0]))
        with pytest.raises(ValueError, match="finite"):
            metric.update(torch.ones(2), torch.tensor([1.0, float("nan")]))

    def test_integer_errors_and_too_few_samples_for_plot(self) -> None:
        metric = AUSE()
        metric.update(torch.tensor([0.2, 0.1]), torch.tensor([2, 1]))
        assert metric.compute().dtype == torch.get_default_dtype()

        metric = AUSE()
        metric.update(torch.tensor([0.2]), torch.tensor([1.0]))
        with pytest.raises(RuntimeError, match="at least two samples"):
            metric.plot()
