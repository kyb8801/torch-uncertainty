import pytest
import torch

from torch_uncertainty.metrics import BrierScore


@pytest.fixture
def vec2d_max() -> torch.Tensor:
    vec = torch.as_tensor([0.5, 0.5])
    return vec.unsqueeze(0)


@pytest.fixture
def vec2d_max_target() -> torch.Tensor:
    vec = torch.as_tensor([0, 1])
    return vec.unsqueeze(0)


@pytest.fixture
def vec2d_max_target1d() -> torch.Tensor:
    return torch.as_tensor([1])


@pytest.fixture
def vec2d_min() -> torch.Tensor:
    vec = torch.as_tensor([0.0, 1.0])
    return vec.unsqueeze(0)


@pytest.fixture
def vec2d_min_target() -> torch.Tensor:
    vec = torch.as_tensor([0, 1])
    return vec.unsqueeze(0)


@pytest.fixture
def vec2d_5classes() -> torch.Tensor:
    return torch.as_tensor([[0.2, 0.6, 0.1, 0.05, 0.05], [0.05, 0.25, 0.1, 0.3, 0.3]])


@pytest.fixture
def vec2d_5classes_target() -> torch.Tensor:
    return torch.as_tensor([[0, 0, 0, 1, 0], [0, 0, 0, 0, 1]])


@pytest.fixture
def vec2d_5classes_target1d() -> torch.Tensor:
    return torch.as_tensor([3, 4])


@pytest.fixture
def vec3d() -> torch.Tensor:
    """Return a torch tensor with a mean BrierScore of 0 and a BrierScore of
    the mean of 0.5 to test the `ensemble` parameter of `BrierScore`.
    """
    vec = torch.as_tensor([[0.0, 1.0], [1.0, 0.0]])
    return vec.unsqueeze(0)


@pytest.fixture
def vec3d_target() -> torch.Tensor:
    vec = torch.as_tensor([0, 1])
    return vec.unsqueeze(0)


@pytest.fixture
def vec3d_target1d() -> torch.Tensor:
    vec = torch.as_tensor([1])
    return vec.unsqueeze(0)


class TestBrierScore:
    """Testing the BrierScore metric class."""

    def test_compute(self, vec2d_min: torch.Tensor, vec2d_min_target: torch.Tensor) -> None:
        metric = BrierScore(num_classes=2)
        metric.update(vec2d_min, vec2d_min_target)
        assert metric.compute() == 0

        metric = BrierScore(num_classes=2, top_class=True)
        metric.update(vec2d_min, vec2d_min_target)
        assert metric.compute() == 0

    def test_compute_max(self, vec2d_max: torch.Tensor, vec2d_max_target: torch.Tensor) -> None:
        metric = BrierScore(num_classes=2, reduction="sum")
        metric.update(vec2d_max, vec2d_max_target)
        assert metric.compute() == 0.5

    def test_compute_max_target1d(
        self, vec2d_max: torch.Tensor, vec2d_max_target1d: torch.Tensor
    ) -> None:
        metric = BrierScore(num_classes=2, reduction="sum")
        metric.update(vec2d_max, vec2d_max_target1d)
        assert metric.compute() == 0.5

    def test_compute_5classes(
        self,
        vec2d_5classes: torch.Tensor,
        vec2d_5classes_target: torch.Tensor,
        vec2d_5classes_target1d: torch.Tensor,
    ) -> None:
        metric = BrierScore(num_classes=5, reduction="sum")
        metric.update(vec2d_5classes, vec2d_5classes_target)
        metric.update(vec2d_5classes, vec2d_5classes_target1d)
        assert (
            metric.compute() / 2
            == 0.2**2 + 0.6**2 + 0.1**2 * 2 + 0.95**2 + 0.05**2 * 2 + 0.25**2 + 0.3**2 + 0.7**2
        )

        metric = BrierScore(num_classes=5, top_class=True, reduction="sum")
        metric.update(vec2d_5classes, vec2d_5classes_target)
        assert metric.compute() == pytest.approx(0.6**2 + 0.3**2)

    def test_multiple_compute_sum(
        self,
        vec2d_min: torch.Tensor,
        vec2d_max: torch.Tensor,
        vec2d_min_target: torch.Tensor,
        vec2d_max_target: torch.Tensor,
    ) -> None:
        metric = BrierScore(num_classes=2, reduction="sum")
        metric.update(vec2d_min, vec2d_min_target)
        metric.update(vec2d_max, vec2d_max_target)
        assert metric.compute() == 0.5

    def test_multiple_compute_mean(
        self,
        vec2d_min: torch.Tensor,
        vec2d_max: torch.Tensor,
        vec2d_min_target: torch.Tensor,
        vec2d_max_target: torch.Tensor,
    ) -> None:
        metric = BrierScore(num_classes=2, reduction="mean")
        metric.update(vec2d_min, vec2d_min_target)
        metric.update(vec2d_max, vec2d_max_target)
        assert metric.compute() == 0.5 / 2

    def test_multiple_compute_none(
        self,
        vec2d_min: torch.Tensor,
        vec2d_max: torch.Tensor,
        vec2d_min_target: torch.Tensor,
        vec2d_max_target: torch.Tensor,
    ) -> None:
        metric = BrierScore(num_classes=2, reduction=None)
        metric.update(vec2d_min, vec2d_min_target)
        metric.update(vec2d_max, vec2d_max_target)
        assert all(metric.compute() == torch.as_tensor([0, 0.5]))

    def test_compute_3d_mean(self, vec3d: torch.Tensor, vec3d_target: torch.Tensor) -> None:
        """Test that the metric returns the mean of the BrierScore over
        the estimators.
        """
        metric = BrierScore(num_classes=2, reduction="mean")
        metric.update(vec3d, vec3d_target)
        assert metric.compute() == 1

    def test_mixed_2d_and_3d_updates(self) -> None:
        metric = BrierScore(num_classes=2, reduction="mean")
        metric.update(torch.tensor([[0.0, 1.0]]), torch.tensor([1]))
        metric.update(
            torch.tensor([[[1.0, 0.0], [1.0, 0.0]]]),
            torch.tensor([1]),
        )

        # The first sample has score 0 and the second has estimator-mean score 2.
        torch.testing.assert_close(metric.compute(), torch.tensor(1.0))

    def test_none_reduction_averages_estimators_per_sample(self) -> None:
        probs = torch.tensor(
            [
                [[0.0, 1.0], [1.0, 0.0]],
                [[0.5, 0.5], [0.5, 0.5]],
            ]
        )
        metric = BrierScore(num_classes=2, reduction="none")
        metric.update(probs, torch.tensor([1, 0]))

        assert metric.compute().shape == (2,)
        torch.testing.assert_close(metric.compute(), torch.tensor([1.0, 0.5]))

    def test_binary_brier_and_top_class(self) -> None:
        probs = torch.tensor([0.1, 0.8])
        target = torch.tensor([0, 1])

        metric = BrierScore(num_classes=1)
        top_metric = BrierScore(num_classes=1, top_class=True)

        torch.testing.assert_close(metric(probs, target), torch.tensor(0.025))
        torch.testing.assert_close(top_metric(probs, target), torch.tensor(0.025))

    def test_compute_3d_sum(self, vec3d: torch.Tensor, vec3d_target: torch.Tensor) -> None:
        metric = BrierScore(num_classes=2, reduction="sum")
        metric.update(vec3d, vec3d_target)
        assert metric.compute() == 1

    def test_compute_3d_sum_target1d(
        self, vec3d: torch.Tensor, vec3d_target1d: torch.Tensor
    ) -> None:
        metric = BrierScore(num_classes=2, reduction="sum")
        metric.update(vec3d, vec3d_target1d)
        assert metric.compute() == 1

    def test_compute_3d_to_2d(self, vec3d: torch.Tensor, vec3d_target: torch.Tensor) -> None:
        metric = BrierScore(num_classes=2, reduction="mean")
        vec3d = vec3d.mean(1)
        metric.update(vec3d, vec3d_target)
        assert metric.compute() == 0.5

    def test_bad_input(self) -> None:
        metric = BrierScore(num_classes=2, reduction="none")
        with pytest.raises(ValueError):
            metric.update(torch.ones(2, 2, 2, 2), torch.ones(2, 2, 2, 2))

    @pytest.mark.parametrize(
        ("probs", "target", "match"),
        [
            (torch.ones(2), torch.zeros(2, dtype=torch.long), "One-dimensional"),
            (torch.ones(2, 3), torch.zeros(2, dtype=torch.long), "Expected 2 classes"),
            (torch.ones(2, 2), torch.zeros(2, 2, 2), "Expected `target`"),
            (torch.ones(2, 2), torch.zeros(3, dtype=torch.long), "same batch size"),
        ],
    )
    def test_invalid_input_shapes(
        self,
        probs: torch.Tensor,
        target: torch.Tensor,
        match: str,
    ) -> None:
        with pytest.raises(ValueError, match=match):
            BrierScore(num_classes=2).update(probs, target)

    def test_binary_column_target_and_ensemble_top_class(self) -> None:
        probs = torch.tensor([[[0.1], [0.3]], [[0.8], [0.6]]])
        target = torch.tensor([[0], [1]])

        metric = BrierScore(num_classes=1, top_class=True, reduction="none")
        metric.update(probs, target)

        torch.testing.assert_close(metric.compute(), torch.tensor([0.05, 0.10]))

    def test_bad_argument(self) -> None:
        with pytest.raises(ValueError, match=r"Expected argument `reduction` to be one of"):
            _ = BrierScore(num_classes=2, reduction="geometric_mean")
