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

        deup = DEUP(task="classification", model=model, n_folds=4, max_epochs=5, device="cpu")
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

        deup = DEUP(task="regression", model=model, n_folds=3, max_epochs=5, device="cpu")
        deup.fit(dl)
        unc = deup(x[:5])
        assert unc.shape == (5,)
        assert torch.all(unc >= 0)

    def test_deup_criterion(self) -> None:
        crit = DEUPCriterion()
        scores = torch.tensor([0.1, 2.0, 0.5])
        assert torch.allclose(crit(scores), scores)

    def test_epistemic_ranks_errors_classification(self) -> None:
        """DEUP uncertainty should correlate with realized CE on a simple setup."""
        torch.manual_seed(2)
        n, in_dim, n_classes = 120, 6, 4
        x = torch.randn(n, in_dim)
        y = torch.randint(0, n_classes, (n,))

        class NoisyLinear(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc = nn.Linear(in_dim, n_classes)

            def forward(self, inputs: torch.Tensor) -> torch.Tensor:
                return self.fc(inputs) + 0.5 * torch.randn_like(self.fc(inputs))

        model = NoisyLinear()
        dl = DataLoader(TensorDataset(x, y), batch_size=20)
        deup = DEUP(task="classification", model=model, n_folds=4, max_epochs=20, device="cpu")
        deup.fit(dl)

        with torch.no_grad():
            logits = model(x)
            ce = nn.CrossEntropyLoss(reduction="none")(logits, y)
            unc = deup(x)
        rho = torch.corrcoef(torch.stack([unc, ce]))[0, 1]
        assert rho.item() > 0.1
