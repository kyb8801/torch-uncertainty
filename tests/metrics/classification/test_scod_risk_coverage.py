import pytest
import torch
from torch import Tensor

from torch_uncertainty.metrics.classification import (
    SCODAUGRC,
    SCODAURC,
    SCODCovAt5Risk,
    SCODCovAtxRisk,
    SCODRiskAt80Cov,
    SCODRiskAtxCov,
)


@pytest.fixture
def scod_inputs() -> tuple[Tensor, Tensor, Tensor]:
    """Return scores yielding SCOD losses [0, 0.25, 0.75, 0].

    With ``ood_cost=0.75``:

    - correct ID costs 0,
    - misclassified ID costs 0.25,
    - OOD costs 0.75.

    The OOD scores are already ordered from most to least acceptable.
    """
    ood_scores = torch.tensor([0.1, 0.2, 0.3, 0.4])
    classification_errors = torch.tensor([False, True, False, False])
    is_ood = torch.tensor([False, False, True, False])
    return ood_scores, classification_errors, is_ood


class TestSCODAURC:
    def test_compute_joint_scod_risk(
        self,
        scod_inputs: tuple[Tensor, Tensor, Tensor],
    ) -> None:
        ood_scores, classification_errors, is_ood = scod_inputs
        metric = SCODAURC(ood_cost=0.75)

        metric.update(
            ood_scores,
            classification_errors,
            is_ood,
        )

        # Accepted losses are [0, 0.25, 0.75, 0], giving cumulative
        # risks [0, 0.125, 1 / 3, 0.25].
        assert torch.allclose(
            metric.partial_compute(),
            torch.tensor([0.0, 0.125, 1 / 3, 0.25]),
        )
        assert metric.compute().item() == pytest.approx(7 / 36)

    def test_compute_across_separate_id_and_ood_updates(self) -> None:
        metric = SCODAURC(ood_cost=0.75)

        metric.update(
            ood_scores=torch.tensor([0.1, 0.2, 0.4]),
            classification_errors=torch.tensor([False, True, False]),
            is_ood=torch.zeros(3, dtype=torch.bool),
        )
        metric.update(
            ood_scores=torch.tensor([0.3]),
            classification_errors=torch.tensor([False]),
            is_ood=torch.ones(1, dtype=torch.bool),
        )

        assert torch.allclose(
            metric.partial_compute(),
            torch.tensor([0.0, 0.125, 1 / 3, 0.25]),
        )
        assert metric.compute().item() == pytest.approx(7 / 36)

    def test_default_cost_extremes(self) -> None:
        ood_scores = torch.tensor([0.1, 0.2, 0.3, 0.4])
        is_id = torch.zeros(4, dtype=torch.bool)

        metric = SCODAURC()
        result = metric(
            ood_scores,
            torch.zeros(4, dtype=torch.bool),
            is_id,
        )
        assert result.item() == pytest.approx(0.0)

        metric = SCODAURC()
        result = metric(
            ood_scores,
            torch.ones(4, dtype=torch.bool),
            is_id,
        )
        assert result.item() == pytest.approx(0.5)

        metric = SCODAURC()
        result = metric(
            ood_scores,
            torch.zeros(4, dtype=torch.bool),
            torch.ones(4, dtype=torch.bool),
        )
        assert result.item() == pytest.approx(0.5)

    def test_classification_error_is_ignored_for_ood(self) -> None:
        metric = SCODAURC(ood_cost=0.75)
        metric.update(
            ood_scores=torch.tensor([0.1, 0.2]),
            classification_errors=torch.tensor([True, True]),
            is_ood=torch.tensor([True, True]),
        )

        assert torch.allclose(
            metric.partial_compute(),
            torch.tensor([0.75, 0.75]),
        )
        assert metric.compute().item() == pytest.approx(0.75)

    def test_single_sample_returns_nan(self) -> None:
        metric = SCODAURC()
        result = metric(
            torch.tensor([0.0]),
            torch.tensor([True]),
            torch.tensor([False]),
        )

        assert result.isnan().all()


class TestSCODAUGRC:
    def test_compute_joint_scod_risk(
        self,
        scod_inputs: tuple[Tensor, Tensor, Tensor],
    ) -> None:
        ood_scores, classification_errors, is_ood = scod_inputs
        metric = SCODAUGRC(ood_cost=0.75)

        result = metric(
            ood_scores,
            classification_errors,
            is_ood,
        )

        assert result.item() == pytest.approx(7 / 48)

    def test_constant_ood_risk(self) -> None:
        metric = SCODAUGRC()
        result = metric(
            ood_scores=torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]),
            classification_errors=torch.zeros(5, dtype=torch.bool),
            is_ood=torch.ones(5, dtype=torch.bool),
        )

        # Constant SCOD risk 0.5 multiplied by coverage before integration.
        assert result.item() == pytest.approx(0.3)


class TestSCODCovAtxRisk:
    def test_compute_maximum_admissible_coverage(
        self,
        scod_inputs: tuple[Tensor, Tensor, Tensor],
    ) -> None:
        ood_scores, classification_errors, is_ood = scod_inputs
        metric = SCODCovAtxRisk(
            risk_threshold=0.13,
            ood_cost=0.75,
        )

        result = metric(
            ood_scores,
            classification_errors,
            is_ood,
        )

        # Risks are [0, 0.125, 1 / 3, 0.25], so the largest
        # admissible coverage is 2 / 4.
        assert result.item() == pytest.approx(0.5)

    def test_compute_fixed_five_percent_risk(
        self,
        scod_inputs: tuple[Tensor, Tensor, Tensor],
    ) -> None:
        ood_scores, classification_errors, is_ood = scod_inputs
        metric = SCODCovAt5Risk(ood_cost=0.75)

        result = metric(
            ood_scores,
            classification_errors,
            is_ood,
        )

        assert result.item() == pytest.approx(0.25)

    def test_no_admissible_coverage_returns_nan(self) -> None:
        metric = SCODCovAtxRisk(
            risk_threshold=0.5,
            ood_cost=0.75,
        )
        result = metric(
            ood_scores=torch.tensor([0.1, 0.2, 0.3]),
            classification_errors=torch.zeros(3, dtype=torch.bool),
            is_ood=torch.ones(3, dtype=torch.bool),
        )

        assert result.isnan().all()


class TestSCODRiskAtxCov:
    def test_compute_risk_at_coverage(
        self,
        scod_inputs: tuple[Tensor, Tensor, Tensor],
    ) -> None:
        ood_scores, classification_errors, is_ood = scod_inputs
        metric = SCODRiskAtxCov(
            cov_threshold=0.5,
            ood_cost=0.75,
        )

        result = metric(
            ood_scores,
            classification_errors,
            is_ood,
        )

        assert result.item() == pytest.approx(0.125)

    def test_compute_fixed_eighty_percent_coverage(
        self,
        scod_inputs: tuple[Tensor, Tensor, Tensor],
    ) -> None:
        ood_scores, classification_errors, is_ood = scod_inputs
        metric = SCODRiskAt80Cov(ood_cost=0.75)

        result = metric(
            ood_scores,
            classification_errors,
            is_ood,
        )

        # ceil(4 * 0.8) = 4 accepted samples.
        assert result.item() == pytest.approx(0.25)


class TestSCODRiskCoverageValidation:
    @pytest.mark.parametrize("score_dtype", [torch.int64, torch.uint8, torch.float16])
    def test_fractional_losses_do_not_follow_score_dtype(
        self,
        score_dtype: torch.dtype,
    ) -> None:
        metric = SCODAURC(ood_cost=0.5)

        metric.update(
            ood_scores=torch.tensor([0, 1], dtype=score_dtype),
            classification_errors=torch.zeros(2, dtype=torch.bool),
            is_ood=torch.ones(2, dtype=torch.bool),
        )

        assert metric.errors[0].dtype == torch.float32
        torch.testing.assert_close(metric.errors[0], torch.tensor([0.5, 0.5]))
        torch.testing.assert_close(metric.partial_compute(), torch.tensor([0.5, 0.5]))

    def test_float64_scores_preserve_precision(self) -> None:
        metric = SCODAURC(ood_cost=0.25)
        metric.update(
            ood_scores=torch.tensor([0.0, 1.0], dtype=torch.float64),
            classification_errors=torch.zeros(2, dtype=torch.bool),
            is_ood=torch.ones(2, dtype=torch.bool),
        )

        assert metric.errors[0].dtype == torch.float64
        assert metric.scores[0].dtype == torch.float64

    @pytest.mark.parametrize("metric_cls", [SCODAURC, SCODAUGRC])
    def test_invalid_ood_cost_type(self, metric_cls: type) -> None:
        with pytest.raises(TypeError, match=r"Expected ood_cost to be of type float"):
            metric_cls(ood_cost=1)

    @pytest.mark.parametrize("ood_cost", [-0.1, 1.1])
    @pytest.mark.parametrize("metric_cls", [SCODAURC, SCODAUGRC])
    def test_invalid_ood_cost_value(
        self,
        metric_cls: type,
        ood_cost: float,
    ) -> None:
        with pytest.raises(ValueError, match=r"ood_cost should be in the range"):
            metric_cls(ood_cost=ood_cost)

    def test_mismatched_input_sizes(self) -> None:
        metric = SCODAURC()

        with pytest.raises(
            ValueError,
            match=r"must contain the same number of elements",
        ):
            metric.update(
                ood_scores=torch.tensor([0.1, 0.2]),
                classification_errors=torch.tensor([False]),
                is_ood=torch.tensor([False, True]),
            )
