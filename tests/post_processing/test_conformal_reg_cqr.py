import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from torch_uncertainty.post_processing import ConformalRegCQR


class TestConformalRegCQR:
    """Testing the ConformalRegCQR class."""

    def test_fit(self) -> None:
        # The model returns its input directly as the [lower, upper] quantiles,
        # so the raw prediction interval is [0, 1] for every calibration sample.
        inputs = torch.tensor([[0.0, 1.0]]).repeat(9, 1)
        targets = torch.tensor([0.5] * 8 + [3.0])

        calibration_set = list(zip(inputs, targets, strict=True))
        dl = DataLoader(calibration_set, batch_size=9)

        conformal = ConformalRegCQR(alpha=0.1, model=nn.Identity())
        conformal.fit(dl)

        # n = 9, alpha = 0.1  ->  level = ceil((n + 1)(1 - alpha)) / n = ceil(9) / 9 = 1.0
        # E_i = max(0 - y_i, y_i - 1) = [-0.5] * 8 + [2.0]  ->  Q = max(E) = 2.0
        assert torch.isclose(conformal.quantile, torch.tensor(2.0))

        out = conformal(inputs)
        assert out.shape == (9, 2)
        expected = torch.tensor([[-2.0, 3.0]]).repeat(9, 1)
        assert torch.allclose(out, expected)

    def test_coverage(self) -> None:
        torch.manual_seed(0)
        n = 2000
        x = torch.rand(n, 1) * 10
        sigma = (0.1 + 0.3 * x / 10).squeeze(-1)
        y = torch.sin(x).squeeze(-1) + sigma * torch.randn(n)

        # A deliberately miscalibrated (too narrow) quantile model.
        center = torch.sin(x).squeeze(-1)
        quantiles = torch.stack([center - 0.2, center + 0.2], dim=-1)

        calibration_set = list(zip(quantiles, y, strict=True))
        dl = DataLoader(calibration_set, batch_size=256)

        conformal = ConformalRegCQR(alpha=0.1, model=nn.Identity())
        conformal.fit(dl)

        interval = conformal(quantiles)
        coverage = ((y >= interval[:, 0]) & (y <= interval[:, 1])).float().mean()
        # CQR restores marginal coverage close to the 1 - alpha target.
        assert coverage >= 0.87

    def test_failures(self) -> None:
        with pytest.raises(RuntimeError):
            _ = ConformalRegCQR(alpha=0.1).quantile
        with pytest.raises(ValueError):
            _ = ConformalRegCQR(alpha=1.5)
