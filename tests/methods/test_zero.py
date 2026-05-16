import pytest
import torch
from torch import nn

from torch_uncertainty.methods import Zero


class TestZero:
    """Testing the Zero wrapper class."""

    @torch.no_grad()
    def test_main(self) -> None:
        model = Zero(nn.Identity(), num_tta=12, filter_views=0.5)
        out = model(torch.randn(2, 10))
        assert out.shape == (2, 10)
        out = model.eval()(torch.randn(12, 3))
        assert out.shape == (1, 3)
        out = model.eval()(torch.randn(24, 3))
        assert out.shape == (2, 3)

    @torch.no_grad()
    def test_tiebreaking(self) -> None:
        # trigger the while loop in eval_forward by creating a tie in votes
        model = Zero(nn.Identity(), num_tta=4, filter_views=0.5)
        model.eval()
        # rows 0,1 are high-confidence (low entropy), rows 2,3 medium-confidence
        # rows 0 and 1 vote for different classes => tie after kept_views=2
        # row 2 breaks the tie
        x = torch.tensor(
            [
                [0.98, 0.01, 0.01],
                [0.01, 0.98, 0.01],
                [0.70, 0.20, 0.10],
                [0.10, 0.70, 0.20],
            ]
        )
        out = model(x)
        assert out.shape == (1, 3)

    def test_failures(self) -> None:
        with pytest.raises(ValueError, match=r"must be in the range"):
            Zero(nn.Identity(), num_tta=12, filter_views=2.1)
        with pytest.raises(
            ValueError, match=r"should be greater than 1/filter_views to use Zero. Got "
        ):
            Zero(nn.Identity(), num_tta=12, filter_views=0.001)
        with pytest.raises(ValueError, match=r"should be strictly positive."):
            Zero(nn.Identity(), num_tta=12, filter_views=1, eps=-1)
