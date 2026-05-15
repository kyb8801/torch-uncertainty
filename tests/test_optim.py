import torch
import torch.nn as nn

from torch_uncertainty.optim import SGHMC, SGLD


class TestSGLD:
    """Testing the SGLD optimizer."""

    def test_step_no_weight_decay(self) -> None:
        model = nn.Linear(4, 2)
        opt = SGLD(model.parameters(), lr=1e-3, noise_factor=1e-2, weight_decay=0)
        x = torch.randn(3, 4)
        loss = nn.functional.mse_loss(model(x), torch.randn(3, 2))
        loss.backward()
        opt.step()

    def test_step_with_weight_decay(self) -> None:
        model = nn.Linear(4, 2)
        opt = SGLD(model.parameters(), lr=1e-3, weight_decay=1e-4)
        x = torch.randn(3, 4)
        loss = nn.functional.mse_loss(model(x), torch.randn(3, 2))
        loss.backward()
        opt.step()

    def test_step_with_closure(self) -> None:
        model = nn.Linear(4, 2)
        opt = SGLD(model.parameters(), lr=1e-3)
        x = torch.randn(3, 4)
        target = torch.randn(3, 2)

        def closure():
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(x), target)
            loss.backward()
            return loss.item()

        result = opt.step(closure=closure)
        assert result is not None

    def test_step_skips_none_grad(self) -> None:
        model = nn.Linear(4, 2)
        opt = SGLD(model.parameters(), lr=1e-3)
        opt.step()


class TestSGHMC:
    """Testing the SGHMC optimizer."""

    def test_step_burn_in(self) -> None:
        model = nn.Linear(4, 2)
        opt = SGHMC(model.parameters(), lr=1e-2, burn_in_steps=5)
        x = torch.randn(3, 4)
        for _ in range(3):
            loss = nn.functional.mse_loss(model(x), torch.randn(3, 2))
            loss.backward()
            opt.step()

    def test_step_after_burn_in(self) -> None:
        model = nn.Linear(4, 2)
        opt = SGHMC(model.parameters(), lr=1e-2, burn_in_steps=2)
        x = torch.randn(3, 4)
        for _ in range(5):
            loss = nn.functional.mse_loss(model(x), torch.randn(3, 2))
            loss.backward()
            opt.step()

    def test_step_with_weight_decay(self) -> None:
        model = nn.Linear(4, 2)
        opt = SGHMC(model.parameters(), lr=1e-2, weight_decay=1e-4)
        x = torch.randn(3, 4)
        loss = nn.functional.mse_loss(model(x), torch.randn(3, 2))
        loss.backward()
        opt.step()

    def test_step_with_closure(self) -> None:
        model = nn.Linear(4, 2)
        opt = SGHMC(model.parameters(), lr=1e-2)
        x = torch.randn(3, 4)
        target = torch.randn(3, 2)

        def closure():
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(x), target)
            loss.backward()
            return loss.item()

        result = opt.step(closure=closure)
        assert result is not None

    def test_step_skips_none_grad(self) -> None:
        model = nn.Linear(4, 2)
        opt = SGHMC(model.parameters(), lr=1e-2)
        opt.step()
