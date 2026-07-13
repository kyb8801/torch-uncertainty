import pytest
import torch

from torch_uncertainty.metrics import (
    IntervalCoverage,
    IntervalScore,
    MeanIntervalWidth,
)


class TestIntervalCoverage:
    """Test the IntervalCoverage (PICP) metric."""

    def test_main(self) -> None:
        lower = torch.tensor([0.0, 1.0, 2.0, 3.0])
        upper = torch.tensor([2.0, 3.0, 4.0, 5.0])
        target = torch.tensor([1.0, 5.0, 3.0, 4.0])  # inside: T, F, T, T
        metric = IntervalCoverage()
        metric.update(lower, upper, target)
        assert metric.compute() == pytest.approx(0.75)

    def test_boundaries_are_inclusive(self) -> None:
        metric = IntervalCoverage()
        metric.update(
            torch.tensor([0.0, 1.0]),
            torch.tensor([2.0, 3.0]),
            torch.tensor([0.0, 3.0]),  # exactly on the lower / upper bound
        )
        assert metric.compute() == pytest.approx(1.0)

    def test_accumulates_across_updates(self) -> None:
        metric = IntervalCoverage()
        metric.update(
            torch.tensor([0.0, 1.0, 2.0, 3.0]),
            torch.tensor([2.0, 3.0, 4.0, 5.0]),
            torch.tensor([1.0, 5.0, 3.0, 4.0]),
        )  # 3 / 4
        metric.update(
            torch.tensor([0.0, 10.0]),
            torch.tensor([1.0, 11.0]),
            torch.tensor([0.5, 0.0]),
        )  # 1 / 2
        assert metric.compute() == pytest.approx(4 / 6)

    def test_multidim_is_flattened(self) -> None:
        metric = IntervalCoverage()
        metric.update(
            torch.zeros(2, 2),
            torch.ones(2, 2),
            torch.tensor([[0.5, 0.5], [2.0, 0.5]]),  # inside: T, T, F, T
        )
        assert metric.compute() == pytest.approx(0.75)

    def test_shape_mismatch(self) -> None:
        metric = IntervalCoverage()
        with pytest.raises(ValueError, match="same shape"):
            metric.update(torch.zeros(3), torch.ones(2), torch.zeros(3))


class TestMeanIntervalWidth:
    """Test the MeanIntervalWidth (MPIW) metric."""

    def test_main(self) -> None:
        metric = MeanIntervalWidth()
        metric.update(
            torch.tensor([0.0, 1.0, 2.0]),
            torch.tensor([2.0, 3.0, 5.0]),  # widths 2, 2, 3
        )
        assert metric.compute() == pytest.approx(7 / 3)

    def test_invalid_interval_order(self) -> None:
        metric = MeanIntervalWidth()
        with pytest.raises(ValueError, match=r"lower.*upper"):
            metric.update(torch.tensor([1.0]), torch.tensor([0.0]))

    def test_shape_mismatch(self) -> None:
        metric = MeanIntervalWidth()
        with pytest.raises(ValueError, match="same shape"):
            metric.update(torch.zeros(3), torch.ones(2))


class TestIntervalScore:
    """Test the IntervalScore (Winkler) metric."""

    def test_inside_equals_width(self) -> None:
        # A target inside the interval is only charged the interval width.
        metric = IntervalScore(coverage=0.8)
        metric.update(torch.tensor([0.0]), torch.tensor([3.0]), torch.tensor([1.5]))
        assert metric.compute() == pytest.approx(3.0)

    def test_above_and_inside(self) -> None:
        # coverage 0.9 -> alpha 0.1 -> penalty factor 2 / 0.1 = 20.
        # pt1 [0, 2], y=1 inside -> 2
        # pt2 [1, 3], y=5 above by 2 -> 2 + 20 * 2 = 42
        # mean = (2 + 42) / 2 = 22
        metric = IntervalScore(coverage=0.9)
        metric.update(
            torch.tensor([0.0, 1.0]),
            torch.tensor([2.0, 3.0]),
            torch.tensor([1.0, 5.0]),
        )
        assert metric.compute() == pytest.approx(22.0)

    def test_below_bound(self) -> None:
        # [2, 4], y=0 below by 2 -> width 2 + 20 * 2 = 42
        metric = IntervalScore(coverage=0.9)
        metric.update(torch.tensor([2.0]), torch.tensor([4.0]), torch.tensor([0.0]))
        assert metric.compute() == pytest.approx(42.0)

    @pytest.mark.parametrize("coverage", [0.0, 1.0, -0.1, 1.5])
    def test_invalid_coverage(self, coverage: float) -> None:
        with pytest.raises(ValueError, match="coverage must be in"):
            IntervalScore(coverage=coverage)

    def test_shape_mismatch(self) -> None:
        metric = IntervalScore(coverage=0.9)
        with pytest.raises(ValueError, match="same shape"):
            metric.update(torch.zeros(3), torch.ones(3), torch.zeros(2))
