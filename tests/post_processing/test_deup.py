"""Tests for DEUP post-processing."""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from tests._dummies.model import dummy_model
from torch_uncertainty.ood_criteria import DEUPCriterion
from torch_uncertainty.post_processing import DEUP


class TestDEUP:
    """Testing the DEUP post-processing class."""

    def test_classification_fit_forward(self) -> None:
        torch.manual_seed(0)
        n, in_dim, n_classes = 80, 4, 3
        x = torch.randn(n, in_dim)
        y = torch.randint(0, n_classes, (n,))
        dl = DataLoader(TensorDataset(x, y), batch_size=16)
        model = dummy_model(in_dim, n_classes)

        deup = DEUP(task="classification", model=model, num_folds=4, max_epochs=5, device="cpu")
        deup.fit(dl)
        unc = deup(x[:8])
        assert unc.shape == (8,)
        assert torch.all(unc >= 0)

        probs = deup.predict_proba(x[:8])
        assert probs.shape == (8, n_classes)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(8), atol=1e-5)

    def test_regression_fit_forward(self) -> None:
        torch.manual_seed(1)
        n, in_dim = 60, 3
        x = torch.randn(n, in_dim)
        y = torch.randn(n, 1)
        dl = DataLoader(TensorDataset(x, y), batch_size=12)
        model = dummy_model(in_dim, 1)

        deup = DEUP(task="regression", model=model, num_folds=3, max_epochs=5, device="cpu")
        deup.fit(dl)
        unc = deup(x[:5])
        assert unc.shape == (5,)
        assert torch.all(unc >= 0)

    def test_deup_criterion(self) -> None:
        crit = DEUPCriterion()
        scores = torch.tensor([0.1, 2.0, 0.5])
        assert torch.allclose(crit(scores), scores)

    def test_epistemic_ranks_errors_classification(self) -> None:
        """DEUP assigns higher uncertainty to regions where the model has higher CE.

        We create two clearly separated groups:
        - Confident group: large positive ``inputs[:, 0]``  → model predicts class 0
          with high probability → labels are class 0 → very low CE (~0.03).
        - Uncertain group: near-zero ``inputs[:, 0]`` → uniform logits → labels are
          wrong classes → CE ≈ log(4) ≈ 1.39.

        The features ``[logits, max_prob, entropy]`` cleanly separate the groups, so
        the error predictor should learn to rank them and DEUP scores should be
        systematically higher for the uncertain group.
        """
        torch.manual_seed(2)
        n, in_dim, n_classes = 100, 4, 4
        half = n // 2

        x = torch.randn(n, in_dim)
        x[:half, 0] = x[:half, 0].abs() + 3.0   # confident: first feature >> 0
        x[half:, 0] = x[half:, 0] * 0.05         # uncertain: first feature ≈ 0

        y = torch.zeros(n, dtype=torch.long)
        y[half:] = torch.randint(1, n_classes, (half,))  # uncertain group: wrong label

        class ConfidenceFromFirstFeature(nn.Module):
            def forward(self, inputs: torch.Tensor) -> torch.Tensor:
                logits = torch.zeros(inputs.shape[0], n_classes)
                logits[:, 0] = inputs[:, 0].clamp(min=0)
                return logits

        model = ConfidenceFromFirstFeature()
        dl = DataLoader(TensorDataset(x, y), batch_size=20)
        deup = DEUP(task="classification", model=model, num_folds=4, max_epochs=50, device="cpu")
        deup.fit(dl)

        with torch.no_grad():
            unc = deup(x)

        assert unc[half:].mean() > unc[:half].mean()
