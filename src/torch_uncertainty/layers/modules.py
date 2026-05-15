from typing import Any

from torch import nn


class Identity(nn.Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """A trivial identity module that returns its inputs unchanged.

        Useful as a placeholder in model construction where a module is
        expected but no operation is required.
        """
        super().__init__()

    def forward(self, *args) -> Any:
        return args
