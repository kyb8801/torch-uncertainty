import pytest
import torch

from torch_uncertainty.metrics.classification import FPR95, FPRx


class TestFPR95:
    """Testing the FPR95 metric class."""

    def test_compute_zero(self) -> None:
        metric = FPR95(pos_label=1)
        metric.update(torch.as_tensor([1] * 99 + [0.99]), torch.as_tensor([1] * 99 + [0]))
        res = metric.compute()
        assert res == 0

    def test_compute_half(self) -> None:
        metric = FPR95(pos_label=1)
        metric.update(
            torch.as_tensor([0.9] * 100 + [0.95] * 50 + [0.85] * 50),
            torch.as_tensor([1] * 100 + [0] * 100),
        )
        res = metric.compute()
        assert res == 0.5

    def test_compute_one(self) -> None:
        metric = FPR95(pos_label=1)
        metric.update(torch.as_tensor([0.99] * 99 + [1]), torch.as_tensor([1] * 99 + [0]))
        res = metric.compute()
        assert res == 1

    def test_compute_nan(self) -> None:
        metric = FPR95(pos_label=1)
        metric.update(torch.as_tensor([0.1] * 50 + [0.4] * 50), torch.as_tensor([0] * 100))
        res = metric.compute()
        assert torch.isnan(res).all()

    def test_error(self) -> None:
        with pytest.raises(ValueError):
            FPRx(recall_level=1.2, pos_label=1)

    def test_tied_scores_respect_minimum_recall(self) -> None:
        scores = torch.tensor([0.9] * 16 + [0.1, 0.1])
        target = torch.tensor([1] * 17 + [0])

        # Reaching 95% recall requires accepting the complete score-0.1 tie,
        # which also accepts the only negative example.
        torch.testing.assert_close(FPR95(pos_label=1)(scores, target), torch.tensor(1.0))
