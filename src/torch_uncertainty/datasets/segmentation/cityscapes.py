from collections.abc import Callable
from typing import Any

import torch
from einops import rearrange
from PIL import Image
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE
from torchvision import tv_tensors
from torchvision.datasets import Cityscapes as TVCityscapes
from torchvision.transforms.v2 import functional as F


class Cityscapes(TVCityscapes):
    color_palette = [
        (128, 64, 128),  # 0: road
        (244, 35, 232),  # 1: sidewalk
        (70, 70, 70),  # 2: building
        (102, 102, 156),  # 3: wall
        (190, 153, 153),  # 4: fence
        (153, 153, 153),  # 5: pole
        (250, 170, 30),  # 6: traffic light
        (220, 220, 0),  # 7: traffic sign
        (107, 142, 35),  # 8: vegetation
        (152, 251, 152),  # 9: terrain
        (70, 130, 180),  # 10: sky
        (220, 20, 60),  # 11: person
        (255, 0, 0),  # 12: rider
        (0, 0, 142),  # 13: car
        (0, 0, 70),  # 14: truck
        (0, 60, 100),  # 15: bus
        (0, 80, 100),  # 16: train
        (0, 0, 230),  # 17: motorcycle
        (119, 11, 32),  # 18: bicycle
        (0, 0, 0),  # 19: void
    ]

    def __init__(
        self,
        root: str,
        split: str = "train",
        mode: str = "fine",
        target_type: list[str] | str = "instance",
        transform: Callable[..., Any] | None = None,
        target_transform: Callable[..., Any] | None = None,
        transforms: Callable[..., Any] | None = None,
    ) -> None:
        """Cityscapes dataset wrapper with train-ID color mapping.

        Extends :class:`torchvision.datasets.Cityscapes` to provide a
        stable color palette for visualization and convenience helpers to
        encode and decode semantic segmentation targets using Cityscapes
        train IDs. A tensor mapping train IDs to RGB colors is constructed
        on initialization for decoding predicted masks.

        Attributes:
            color_palette: List of RGB tuples for each class label in Cityscapes.
            train_id_to_color: Tensor mapping train IDs to RGB colors for decoding.

        Args:
            root: Root directory of the Cityscapes dataset.
            split: Dataset split to use (e.g., "train", "val", or "test"). Defaults to ``"train"``.
            mode: Annotation mode ("fine" or "coarse"). Defaults to ``"fine"``.
            target_type: One or more target types to load ("instance", "semantic", etc.). Defaults to ``"instance"``.
            transform: Transformation applied to the input image. Defaults to ``None``.
            target_transform: Transformation applied to the target. Defaults to ``None``.
            transforms: Combined transformation for image and target. Defaults to ``None``.

        """
        super().__init__(
            root,
            split,
            mode,
            target_type,
            transform,
            target_transform,
            transforms,
        )
        train_id_to_color = [
            c.color for c in self.classes if (c.train_id != -1 and c.train_id != 255)
        ]
        train_id_to_color.append([0, 0, 0])
        self.train_id_to_color = torch.tensor(train_id_to_color)

    @classmethod
    def encode_target(cls, target: Image.Image) -> Image.Image:
        """Encode a Cityscapes target PIL image into a train-ID image.

        The input is a color-coded PIL image (train-ID or label colors). The
        method converts it to a single-channel image where each pixel value
        is the corresponding train ID.

        Args:
            target: A PIL image containing the ground-truth target map.

        Returns:
            A PIL image where pixel values correspond to Cityscapes train IDs.
        """
        colored_target = F.pil_to_tensor(target)
        colored_target = rearrange(colored_target, "c h w -> h w c")
        target = torch.zeros_like(colored_target[..., :1])
        # convert target color to index
        for cityscapes_class in cls.classes:
            target[
                (colored_target == torch.tensor(cityscapes_class.id, dtype=target.dtype)).all(
                    dim=-1
                )
            ] = cityscapes_class.train_id

        return F.to_pil_image(rearrange(target, "h w c -> c h w"))

    def decode_target(self, target: torch.Tensor) -> torch.Tensor:
        """Decode a train-ID tensor into an RGB tensor using the palette.

        Pixels with value ``255`` are treated as void and mapped to the
        last palette entry (black). The returned tensor contains RGB color
        values for each pixel according to the dataset palette.

        Args:
            target: Integer tensor of train IDs.

        Returns:
            A tensor of RGB colors with shape ``(H, W, 3)`` mapped from train IDs.
        """
        target[target == 255] = -1
        return self.train_id_to_color[target]

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        """Return the sample at the given index.

        Args:
            index: Integer index of the sample to retrieve.

        Returns:
            ``(image, target)`` tuple: If ``target_type`` contains multiple
            types, ``target`` is a tuple with each corresponding target. If
            ``target_type=="polygon"``, the target is a JSON object; for
            semantic targets the returned object is a mask or ``tv_tensors.Mask``.
        """
        image = tv_tensors.Image(Image.open(self.images[index]).convert("RGB"))

        targets: Any = []
        for i, t in enumerate(self.target_type):
            if t == "polygon":
                target = self._load_json(self.targets[index][i])
            elif t == "semantic":
                target = tv_tensors.Mask(self.encode_target(Image.open(self.targets[index][i])))
            else:
                target = Image.open(self.targets[index][i])

            targets.append(target)

        target = tuple(targets) if len(targets) > 1 else targets[0]

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def plot_sample(self, index: int, ax: _AX_TYPE | None = None) -> _PLOT_OUT_TYPE:
        """Plot a dataset sample (image and decoded target) for inspection.

        Intended behavior: load the sample at ``index``, decode the target
        to RGB using :attr:`train_id_to_color`, and render the image and
        overlay/target on the provided axis. If ``ax`` is ``None``, a new
        matplotlib figure and axis should be created.

        Args:
            index: Index of the sample to plot.
            ax: Optional matplotlib axis to draw on. Defaults to ``None``.

        Returns:
            A tuple ``(fig, ax)`` or the axis object used for plotting, depending on
            the plotting utility in use.

        Raises:
            NotImplementedError: This plotting helper is not implemented yet.
        """
        raise NotImplementedError("plot_sample is not implemented yet.")
