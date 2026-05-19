import torch
from torch import Tensor
from torch.nn import LayerNorm

from .utils import ChannelBack, ChannelFront


class ChannelLayerNorm(LayerNorm):
    def __init__(
        self,
        normalized_shape: int | list[int],
        eps: float = 0.00001,
        elementwise_affine: bool = True,
        bias: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | str | None = None,
    ) -> None:
        """Channel-wise Layer Normalization.

        Args:
            normalized_shape: Input shape from an expected input of size
                ``(N, C, ...)``. Defaults to provided value.
            eps: A value added to the denominator for numerical stability. Defaults to ``1e-5``.
            elementwise_affine: If ``True``, this module has learnable per-channel affine parameters. Defaults to ``True``.
            bias: Included for API consistency; LayerNorm controls affine via ``elementwise_affine``. Defaults to ``True``.
            device: Device for parameters. Defaults to ``None``.
            dtype: Data type for parameters. Defaults to ``None``.
        """
        super().__init__(normalized_shape, eps, elementwise_affine, bias, device, dtype)
        self.cback = ChannelBack()
        self.cfront = ChannelFront()

    def forward(self, input: Tensor) -> Tensor:  # noqa: A002
        return self.cfront(super().forward(self.cback(input)))
