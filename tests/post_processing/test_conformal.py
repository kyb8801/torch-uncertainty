import pytest
import torch
from einops import repeat
from torch import nn
from torch.utils.data import DataLoader

from torch_uncertainty.post_processing import (
    Conformal,
    ConformalClsAPS,
    ConformalClsRAPS,
    ConformalClsTHR,
)


class TestConformal:
    """Testing the Conformal class."""

    def test_errors(self) -> None:
        Conformal.__abstractmethods__ = set()
        conformal = Conformal(
            model=None,
            alpha=0.1,
            ts_init_val=1,
            ts_lr=1,
            ts_max_iter=1,
            enable_ts=True,
            device="cpu",
        )
        assert conformal.model.model is None
        conformal.set_model(nn.Identity())
        assert isinstance(conformal.model.model, nn.Identity)
        conformal.fit(None)
        conformal.forward(None)
        conformal.conformal(None)


class TestConformalClsAPS:
    """Testing the ConformalClsRAPS class."""

    def test_fit(self) -> None:
        inputs = repeat(torch.tensor([0.6, 0.3, 0.1]), "c -> b c", b=10).log()
        labels = torch.tensor([0, 2] + [1] * 8)

        calibration_set = list(zip(inputs, labels, strict=True))
        dl = DataLoader(calibration_set, batch_size=10)

        conformal = ConformalClsAPS(alpha=0.1, model=nn.Identity(), randomized=False)
        conformal.fit(dl)
        out = conformal.conformal(inputs)
        assert out.shape == (10, 3)
        assert (
            out == repeat(torch.tensor([True, True, False]), "c -> b c", b=10).float() / 2
        ).all()

        conformal = ConformalClsAPS(
            alpha=0.1, model=nn.Identity(), randomized=True, enable_ts=False
        )
        conformal.fit(dl)
        out = conformal.conformal(inputs)
        assert out.shape == (10, 3)

    def test_failures(self) -> None:
        with pytest.raises(RuntimeError):
            _ = ConformalClsAPS(
                alpha=0.1,
            ).quantile


class TestConformalClsRAPS:
    """Testing the ConformalClsRAPS class."""

    def test_fit(self) -> None:
        inputs = repeat(torch.tensor([6.0, 4.0, 1.0]), "c -> b c", b=10)
        labels = torch.tensor([0, 2] + [1] * 8)

        calibration_set = list(zip(inputs, labels, strict=True))
        dl = DataLoader(calibration_set, batch_size=10)

        conformal = ConformalClsRAPS(alpha=0.1, model=nn.Identity(), randomized=False)
        conformal.set_model(nn.Identity())
        conformal.fit(dl)
        out = conformal.conformal(inputs)
        assert out.shape == (10, 3)
        assert (
            out == repeat(torch.tensor([True, True, False]), "c -> b c", b=10).float() / 2
        ).all()

        conformal = ConformalClsRAPS(
            alpha=0.1, model=nn.Identity(), randomized=True, enable_ts=False
        )
        conformal.fit(dl)
        out = conformal.conformal(inputs)
        assert out.shape == (10, 3)

    def test_quantile_before_fit(self) -> None:
        conformal = ConformalClsRAPS(alpha=0.1)

        with pytest.raises(
            RuntimeError,
            match=r"Quantile q_hat is not set\. Run `\.fit\(\)` first\.",
        ):
            _ = conformal.quantile

    @pytest.mark.parametrize(
        "penalty",
        [
            None,
            "0.1",
            torch.tensor(0.1),
        ],
    )
    def test_invalid_penalty_type(self, penalty: object) -> None:
        with pytest.raises(
            TypeError,
            match=r"penalty should be a float or integer",
        ):
            ConformalClsRAPS(alpha=0.1, penalty=penalty)

    @pytest.mark.parametrize(
        "penalty",
        [
            float("nan"),
            float("inf"),
            float("-inf"),
        ],
    )
    def test_non_finite_penalty(self, penalty: float) -> None:
        with pytest.raises(
            ValueError,
            match=r"penalty should be finite",
        ):
            ConformalClsRAPS(alpha=0.1, penalty=penalty)

    @pytest.mark.parametrize("penalty", [-0.1, -1])
    def test_negative_penalty(self, penalty: float) -> None:
        with pytest.raises(
            ValueError,
            match=r"penalty should be non-negative",
        ):
            ConformalClsRAPS(alpha=0.1, penalty=penalty)

    @pytest.mark.parametrize(
        "regularization_rank",
        [
            None,
            "1",
            0.1,
            1.0,
        ],
    )
    def test_invalid_regularization_rank_type(
        self,
        regularization_rank: object,
    ) -> None:
        with pytest.raises(
            TypeError,
            match=r"regularization_rank should be an integer",
        ):
            ConformalClsRAPS(
                alpha=0.1,
                regularization_rank=regularization_rank,
            )

    @pytest.mark.parametrize("regularization_rank", [0, -1])
    def test_non_positive_regularization_rank(
        self,
        regularization_rank: int,
    ) -> None:
        with pytest.raises(
            ValueError,
            match=r"regularization_rank should be strictly positive",
        ):
            ConformalClsRAPS(
                alpha=0.1,
                regularization_rank=regularization_rank,
            )

    def test_temperature_when_disabled(self) -> None:
        conformal = ConformalClsRAPS(
            alpha=0.1,
            model=nn.Identity(),
            enable_ts=False,
        )

        with pytest.raises(
            RuntimeError,
            match=r"Cannot return temperature when enable_ts is False\.",
        ):
            _ = conformal.temperature


@pytest.mark.parametrize(
    ("conformal_cls", "kwargs"),
    [
        pytest.param(ConformalClsAPS, {}, id="aps"),
        pytest.param(ConformalClsRAPS, {"penalty": 0.0}, id="raps"),
    ],
)
def test_empty_prediction_set_fallback(
    conformal_cls: type[ConformalClsAPS],
    kwargs: dict,
) -> None:
    # The true calibration label is ranked second, producing q_hat = 0.8.
    calibration_probs = repeat(
        torch.tensor([0.55, 0.25, 0.20]),
        "c -> b c",
        b=10,
    )
    calibration_labels = torch.ones(10, dtype=torch.long)
    calibration_loader = DataLoader(
        list(zip(calibration_probs.log(), calibration_labels, strict=True)),
        batch_size=10,
    )

    conformal = conformal_cls(
        alpha=0.1,
        model=nn.Identity(),
        randomized=False,
        enable_ts=False,
        **kwargs,
    )
    conformal.fit(calibration_loader)

    test_probs = torch.tensor(
        [
            [0.980, 0.015, 0.005],  # Minimum score is 0.98: empty raw set.
            [0.550, 0.250, 0.200],  # Two classes are below q_hat.
        ]
    )
    test_logits = test_probs.log()

    scores = conformal._calculate_all_labels(conformal.model_forward(test_logits))
    raw_prediction_set = scores <= conformal.quantile

    # Verify that both branches of the fallback are genuinely exercised.
    assert torch.equal(
        raw_prediction_set,
        torch.tensor(
            [
                [False, False, False],
                [True, True, False],
            ]
        ),
    )

    output = conformal.conformal(test_logits)

    torch.testing.assert_close(
        output,
        torch.tensor(
            [
                [1.0, 0.0, 0.0],  # Lowest-score singleton fallback.
                [0.5, 0.5, 0.0],  # Existing non-empty set is unchanged.
            ]
        ),
    )
    assert torch.isfinite(output).all()
    torch.testing.assert_close(
        output.sum(dim=-1),
        torch.ones(output.shape[0]),
    )


class TestConformalClsTHR:
    """Testing the ConformalClsTHR class."""

    def test_main(self) -> None:
        conformal = ConformalClsTHR(alpha=0.1, model=None, ts_init_val=2)
        assert conformal.temperature == 2.0
        conformal.set_model(nn.Identity())
        assert isinstance(conformal.model.model, nn.Identity)

    def test_fit(self) -> None:
        inputs = repeat(torch.tensor([0.6, 0.3, 0.1]), "c -> b c", b=10).log()
        labels = torch.tensor([0, 2] + [1] * 8)

        calibration_set = list(zip(inputs, labels, strict=True))
        dl = DataLoader(calibration_set, batch_size=10)

        conformal = ConformalClsTHR(
            alpha=0.1, model=nn.Identity(), ts_init_val=1, ts_lr=1, ts_max_iter=10, enable_ts=True
        )
        conformal.fit(dl)
        out = conformal.conformal(inputs)
        assert out.shape == (10, 3)
        assert (
            out == repeat(torch.tensor([True, True, False]), "c -> b c", b=10).float() / 2
        ).all()

        conformal = ConformalClsTHR(alpha=0.1, model=nn.Identity(), enable_ts=False)
        conformal.fit(dl)

    def test_failures(self) -> None:
        with pytest.raises(RuntimeError):
            _ = ConformalClsTHR(
                alpha=0.1,
            ).quantile
