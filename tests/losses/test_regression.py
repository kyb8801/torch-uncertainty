import math

import pytest
import torch
from torch.distributions import Independent, Normal

from torch_uncertainty.losses import (
    BetaNLL,
    DERLoss,
    DistributionNLLLoss,
    PinballLoss,
)
from torch_uncertainty.utils.distributions import NormalInverseGamma


class TestDistributionNLL:
    """Testing the DistributionNLLLoss class."""

    def test_sum(self) -> None:
        loss = DistributionNLLLoss(reduction="sum")
        dist = Normal(0, 1)
        loss(dist, torch.tensor([0.0]))


class TestDERLoss:
    """Testing the DERLoss class."""

    def test_main(self) -> None:
        loss = DERLoss(reg_weight=1e-2)
        layer = NormalInverseGamma
        inputs = layer(torch.ones(1), torch.ones(1), torch.ones(1), torch.ones(1))
        targets = torch.tensor([[1.0]], dtype=torch.float32)

        assert loss(inputs, targets) == pytest.approx(2 * math.log(2))

        loss = DERLoss(
            reg_weight=1e-2,
            reduction="sum",
        )
        inputs = layer(
            torch.ones((2, 1)),
            torch.ones((2, 1)),
            torch.ones((2, 1)),
            torch.ones((2, 1)),
        )

        inputs = Independent(inputs, 0)

        assert loss(
            inputs,
            targets,
        ) == pytest.approx(4 * math.log(2))

        loss = DERLoss(
            reg_weight=1e-2,
            reduction="none",
        )

        assert loss(
            inputs,
            targets,
        ) == pytest.approx([2 * math.log(2), 2 * math.log(2)])

    def test_failures(self) -> None:
        with pytest.raises(
            ValueError,
            match=r"The regularization weight should be non-negative, but got ",
        ):
            DERLoss(reg_weight=-1)

        with pytest.raises(ValueError, match=r"is not a valid value for reduction."):
            DERLoss(reg_weight=1.0, reduction="median")


class TestBetaNLL:
    """Testing the BetaNLL class."""

    def test_main(self) -> None:
        loss = BetaNLL(beta=0.5)

        inputs = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
        targets = torch.tensor([[1.0]], dtype=torch.float32)

        assert loss(*inputs.split(1, dim=-1), targets) == 0

        loss = BetaNLL(
            beta=0.5,
            reduction="sum",
        )

        assert (
            loss(
                *inputs.repeat(2, 1).split(1, dim=-1),
                targets.repeat(2, 1),
            )
            == 0
        )

        loss = BetaNLL(
            beta=0.5,
            reduction="none",
        )

        assert loss(
            *inputs.repeat(2, 1).split(1, dim=-1),
            targets.repeat(2, 1),
        ) == pytest.approx([0.0, 0.0])

    def test_failures(self) -> None:
        with pytest.raises(ValueError, match=r"The beta parameter should be in range "):
            BetaNLL(beta=-1)

        with pytest.raises(ValueError, match=r"is not a valid value for reduction."):
            BetaNLL(beta=1.0, reduction="median")


class TestPinballLoss:
    """Testing the PinballLoss class."""

    def test_main(self) -> None:
        # At τ=0.5 and perfect prediction the loss is zero.
        loss = PinballLoss(quantile=0.5)
        pred = torch.tensor([1.0])
        target = torch.tensor([1.0])
        assert loss(pred, target) == pytest.approx(0.0)

        # tau 0.5 is MAE/2: underestimate and overestimate are symmetric.
        pred_low = torch.tensor([0.0])
        pred_high = torch.tensor([1.0])
        target_one = torch.tensor([1.0])
        target_zero = torch.tensor([0.0])
        assert loss(pred_low, target_one) == pytest.approx(0.5)
        assert loss(pred_high, target_zero) == pytest.approx(0.5)

        # tau 0.9 penalises underestimation more heavily than overestimation.
        loss_q90 = PinballLoss(quantile=0.9)
        assert loss_q90(pred_low, target_one) == pytest.approx(0.9)
        assert loss_q90(pred_high, target_zero) == pytest.approx(0.1)

        # reduction "sum"
        loss_sum = PinballLoss(quantile=0.5, reduction="sum")
        preds = torch.tensor([0.0, 1.0])
        targets = torch.tensor([1.0, 0.0])
        assert loss_sum(preds, targets) == pytest.approx(1.0)

        # reduction "none"
        loss_none = PinballLoss(quantile=0.5, reduction="none")
        result = loss_none(preds, targets)
        assert result.tolist() == pytest.approx([0.5, 0.5])

    def test_failures(self) -> None:
        with pytest.raises(
            ValueError,
            match=r"The quantile parameter should be in \(0, 1\)",
        ):
            PinballLoss(quantile=0.0)

        with pytest.raises(
            ValueError,
            match=r"The quantile parameter should be in \(0, 1\)",
        ):
            PinballLoss(quantile=1.0)

        with pytest.raises(ValueError, match=r"is not a valid value for reduction."):
            PinballLoss(quantile=0.5, reduction="median")
