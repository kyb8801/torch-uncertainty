"""Modified from https://github.com/nikitadurasov/masksembles/."""

from typing import Any, Literal

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn.common_types import _size_2_t


def _generate_masks(m: int, n: int, s: float) -> np.ndarray:
    """Generates set of binary masks with properties defined by n, m, s params.
    Results of this function are stochastic: repeated calls with the same
    arguments may produce different outputs. Use :func:`generate_masks` and
    :func:`generation_wrapper` for more deterministic behaviour.

    Args:
        m: Number of ones in each mask.
        n: Number of masks in the set.
        s: Scale parameter controlling overlap of generated masks.

    Returns:
        np.ndarray: Matrix of binary vectors.
    """
    rng = np.random.default_rng()
    total_positions = int(m * s)
    masks = []

    for _ in range(n):
        new_vector = np.zeros([total_positions])
        idx = rng.choice(range(total_positions), m, replace=False)
        new_vector[idx] = 1
        masks.append(new_vector)

    masks = np.array(masks)
    # drop useless positions
    return masks[:, ~np.all(masks == 0, axis=0)]


def generate_masks(m: int, n: int, s: float) -> np.ndarray:
    """Generate a set of binary masks with properties defined by ``m``, ``n``, ``s``.

    The function repeatedly calls :func:`_generate_masks` until the resulting
    masks match the expected feature size. The process is stochastic, so
    repetition ensures a correct-sized output.

    Args:
        m: Number of ones in each mask.
        n: Number of masks in the set.
        s: Scale parameter controlling overlap of generated masks.

    Returns:
        np.ndarray: Matrix of binary vectors.
    """
    masks = _generate_masks(m, n, s)
    # hardcoded formula for expected size, check reference
    expected_size = int(m * s * (1 - (1 - 1 / s) ** n))
    while masks.shape[1] != expected_size:
        masks = _generate_masks(m, n, s)
    return masks


def generation_wrapper(c: int, n: int, scale: float) -> np.ndarray:
    """Generate binary masks with a target number of channel features.

    This wrapper produces masks with an expected number of active features
    equal to ``c``. It is convenient for torch-like layers where tensor
    shapes must be known in advance.

    Args:
        c: Number of channels in generated masks.
        n: Number of masks in the set.
        scale: Scale parameter controlling overlap of generated masks.

    Raises:
        ValueError: If ``c < 10``.
        ValueError: If ``scale > 6.0``.

    Returns:
        np.ndarray: matrix of binary vectors
    """
    if c < 10:
        raise ValueError(
            "Masksembles cannot be used when the number of channels is less than 10. "
            f"Current value is (channels={c}). Increase the number of features "
            "or remove this Masksembles instance from your architecture."
        )

    if scale > 6.0:
        raise ValueError(
            "Masksembles cannot be used when the scale parameter is larger than 6. "
            f"Current value is (scale={scale})."
        )

    # inverse formula for number of active features in masks
    active_features = int(int(c) / (scale * (1 - (1 - 1 / scale) ** n)))

    # Use binary search to find the correct value of the scale
    masks = generate_masks(active_features, n, scale)
    up = 4 * scale
    down = max(0.2 * scale, 1.0)
    s = (down + up) / 2
    im_s = -1
    while im_s != c:
        masks = generate_masks(active_features, n, s)
        im_s = masks.shape[-1]
        if im_s < c:
            down = s
            s = (down + up) / 2
        elif im_s > c:
            up = s
            s = (down + up) / 2

    return masks


class Mask1d(nn.Module):
    def __init__(self, channels: int, num_masks: int, scale: float, **factory_kwargs) -> None:
        super().__init__()
        self.num_masks = num_masks

        masks = generation_wrapper(channels, num_masks, scale)
        masks = torch.from_numpy(masks)
        self.masks = torch.nn.Parameter(masks, requires_grad=False).to(
            device=factory_kwargs["device"]
        )

    def forward(self, inputs: Tensor) -> Tensor:
        batch = inputs.shape[0]
        x = torch.split(inputs.unsqueeze(1), batch // self.num_masks, dim=0)
        x = torch.cat(x, dim=1).permute([1, 0, 2])
        x = x * self.masks.unsqueeze(1)
        x = torch.cat(torch.split(x, 1, dim=0), dim=1)
        return torch.as_tensor(x, dtype=inputs.dtype).squeeze(0)


class Mask2d(nn.Module):
    def __init__(self, channels: int, num_masks: int, scale: float, **factory_kwargs) -> None:
        super().__init__()
        self.num_masks = num_masks

        masks = generation_wrapper(channels, num_masks, scale)
        masks = torch.from_numpy(masks)
        self.masks = torch.nn.Parameter(masks, requires_grad=False).to(
            device=factory_kwargs["device"]
        )

    def forward(self, inputs: Tensor) -> Tensor:
        batch = inputs.shape[0]
        x = torch.split(inputs.unsqueeze(1), batch // self.num_masks, dim=0)
        x = torch.cat(x, dim=1).permute([1, 0, 2, 3, 4])
        x = x * self.masks.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        x = torch.cat(torch.split(x, 1, dim=0), dim=1)
        return torch.as_tensor(x, dtype=inputs.dtype).squeeze(0)


class MaskedLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_estimators: int,
        scale: float,
        bias: bool = True,
        device: Any | None = None,
        dtype: Any | None = None,
    ) -> None:
        r"""Masksembles-style Linear layer.

        This layer computes fully-connected operation for a given number of
        estimators (:attr:`num_estimators`) with a given :attr:`scale`.

        Args:
            in_features: Number of input features of the linear layer.
            out_features: Number of channels produced by the linear layer.
            num_estimators: The number of estimators grouped in the layer.
            scale: The scale parameter for the masks.
            bias: It ``True``, adds a learnable bias to the output. Defaults to ``True``.
            groups: Number of blocked connections from input channels to output channels. Defaults to ``1``.
            device: The desired device of returned tensor. Defaults to ``None``.
            dtype: The desired data type of returned tensor. Defaults to ``None``.

        Warning:
            Be sure to apply a repeat on the batch at the start of the training
            if you use `MaskedLinear`.

        References:
            [1] `Masksembles for Uncertainty Estimation, Nikita Durasov, Timur Bagautdinov, Pierre Baque, Pascal Fua
            <https://arxiv.org/abs/2012.08334>`_.

        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        if scale is None:
            raise ValueError("You must specify the value of the arg. `scale`")
        if scale < 1:
            raise ValueError(f"Attribute `scale` should be >= 1, not {scale}.")

        self.mask = Mask1d(in_features, num_masks=num_estimators, scale=scale, **factory_kwargs)
        self.linear = nn.Linear(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            **factory_kwargs,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.linear(self.mask(inputs))


class MaskedConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        num_estimators: int,
        scale: float,
        stride: _size_2_t = 1,
        padding: str | _size_2_t = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        device: Any | None = None,
        dtype: Any | None = None,
    ) -> None:
        r"""Masksembles-style Conv2d layer.

        Args:
            in_channels: Number of channels in the input image.
            out_channels: Number of channels produced by the convolution.
            kernel_size: Size of the convolving kernel.
            num_estimators: Number of estimators in the ensemble.
            scale: The scale parameter for the masks.
            stride: Stride of the convolution. Defaults to ``1``.
            padding (int, tuple or str): Padding added to all four sides of the input. Defaults to ``0``.
            dilation: Spacing between kernel elements. Defaults to ``1``.
            groups: Number of blocked connexions from input channels to output channels for each estimator. Defaults to ``1``.
            bias: If ``True``, adds a learnable bias to the output. Defaults to ``True``.
            device: The desired device of returned tensor. Defaults to ``None``.
            dtype: The desired data type of returned tensor. Defaults to ``None``.

        Warning:
            Be sure to apply a repeat on the batch at the start of the training
            if you use `MaskedConv2d`.

        References:
            [1] `Masksembles for Uncertainty Estimation, Nikita Durasov, Timur Bagautdinov, Pierre Baque, Pascal Fua
            <https://arxiv.org/abs/2012.08334>`_.


        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        if scale is None:
            raise ValueError("You must specify the value of the arg. `scale`")
        if scale < 1:
            raise ValueError(f"Attribute `scale` should be >= 1, not {scale}.")

        self.mask = Mask2d(in_channels, num_masks=num_estimators, scale=scale, **factory_kwargs)
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode="zeros",
            **factory_kwargs,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(self.mask(inputs))


class MaskedConvTranspose2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        num_estimators: int,
        scale: float,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
        output_padding: _size_2_t = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: _size_2_t = 1,
        padding_mode: Literal["circular", "reflect", "replicate", "zeros"] = "zeros",
        device: Any | None = None,
        dtype: Any | None = None,
    ) -> None:
        r"""Masksembles-style ConvTranspose2d layer.

        Args:
            in_channels: Number of channels in the input image.
            out_channels: Number of channels produced by the convolution.
            kernel_size: Size of the convolving kernel.
            num_estimators: Number of estimators in the ensemble.
            scale: The scale parameter for the masks.
            stride: Stride of the convolution. Defaults to ``1``.
            padding: Padding added to all four sides of the input. Defaults to ``0``.
            output_padding (int, tuple or str): Additional size added to one side of each dimension in the output shape. Defaults to ``0``.
            groups: Number of blocked connexions from input channels to output channels for each estimator. Defaults to ``1``.
            bias: If ``True``, adds a learnable bias to the output. Defaults to ``True``.
            dilation: Spacing between kernel elements. Defaults to ``1``.
            padding_mode: _description_. Defaults to ``'zeros'``.
            device: The desired device of returned tensor. Defaults to ``None``.
            dtype: The desired data type of returned tensor. Defaults to ``None``.

        Warning:
            Be sure to apply a repeat on the batch at the start of the training
            if you use `MaskedConvTranspose2d`.

        References:
            [1] `Masksembles for Uncertainty Estimation, Nikita Durasov, Timur Bagautdinov, Pierre Baque, Pascal Fua
            <https://arxiv.org/abs/2012.08334>`_
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if scale is None:
            raise ValueError("You must specify the value of the arg. `scale`")
        if scale < 1:
            raise ValueError(f"Attribute `scale` should be >= 1, not {scale}.")
        self.mask = Mask2d(in_channels, num_masks=num_estimators, scale=scale, **factory_kwargs)
        self.conv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
            dilation=dilation,
            padding_mode=padding_mode,
            **factory_kwargs,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(self.mask(inputs))
