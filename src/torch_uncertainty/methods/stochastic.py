import torch
from torch import Tensor, nn

from torch_uncertainty.layers.bayesian import bayesian_modules


class StochasticModel(nn.Module):
    def __init__(
        self,
        core_model: nn.Module,
        num_samples: int,
        probabilistic: bool = False,
    ) -> None:
        super().__init__()
        self.core_model = core_model
        self.num_samples = num_samples
        self.probabilistic = probabilistic

    def eval_forward(self, x: Tensor) -> Tensor | dict[str, Tensor]:
        out = [self.core_model(x) for _ in range(self.num_samples)]
        if self.probabilistic:
            key_set = {tuple(o.keys()) for o in out}
            return {k: torch.cat([o[k] for o in out], dim=0) for k in key_set.pop()}
        return torch.cat(out, dim=0)

    def forward(self, x: Tensor) -> Tensor | dict[str, Tensor]:
        if self.training:
            return self.core_model.forward(x)
        return self.eval_forward(x)

    def _inner_sample(self, module: nn.Module) -> tuple:  # coverage: ignore
        """Check that the module has a sampling method, then sample it.

        Args:
            module: The module to sample.

        Raises:
            TypeError: Triggered when the module doesn't implement sample.
            TypeError: Triggered when the module sampling function does not return a
                tuple.

        Returns:
            tuple: The weight and biases. Should be both torch Tensors.
        """
        sample_fn = getattr(module, "sample", None)
        if not callable(sample_fn):
            raise TypeError("Bayesian module must implement `sample()`.")
        weight_bias = sample_fn()
        if not isinstance(weight_bias, tuple) or len(weight_bias) != 2:
            raise TypeError("`sample()` must return (weight, bias).")
        return weight_bias

    def sample(self, num_samples: int = 1) -> list[dict[str, Tensor]]:
        """Sample the wrapped model multiple times.

        Args:
            num_samples: Number of samples to generate. Defaults to ``1``.

        Returns:
            list[dict[str, Tensor]]: Sampled model states.
        """
        sampled_models = [{} for _ in range(num_samples)]
        for module_name in self.core_model._modules:
            module = self.core_model._modules[module_name]
            if module is None:  # coverage: ignore
                continue
            if isinstance(module, bayesian_modules):
                for model in sampled_models:
                    weight, bias = self._inner_sample(module)
                    model[module_name + ".weight"] = weight
                    if bias is not None:  # coverage: ignore
                        model[module_name + ".bias"] = bias
            else:
                for model in sampled_models:
                    state = module.state_dict()
                    if not len(state):  # no parameter
                        break
                    # TODO: fix this
                    model |= {
                        module_name + "." + key: val for key, val in module.state_dict().items()
                    }
        return sampled_models

    def freeze(self) -> None:
        """Freeze all Bayesian submodules in the wrapped model."""
        for module in self.core_model.modules():
            if isinstance(module, bayesian_modules):
                freeze_fn = getattr(module, "freeze", None)
                if callable(freeze_fn):
                    freeze_fn()

    def unfreeze(self) -> None:
        """Unfreeze all Bayesian submodules in the wrapped model."""
        for module in self.core_model.modules():
            if isinstance(module, bayesian_modules):
                unfreeze_fn = getattr(module, "unfreeze", None)
                if callable(unfreeze_fn):
                    unfreeze_fn()
