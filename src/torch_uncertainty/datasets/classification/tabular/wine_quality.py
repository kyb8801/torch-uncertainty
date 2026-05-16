from collections.abc import Callable
from pathlib import Path

import pandas as pd
import torch

from .base import TabularClassificationDataset


class WineQuality(TabularClassificationDataset):
    """The UCI Wine Quality dataset.

    Predicts wine quality from physicochemical measurements. Supports both
    red and white wine variants. In binary mode the quality score is
    thresholded; in multi-class mode the raw integer scores (3-9) are kept.

    Reference:
        P. Cortez et al., *Modeling wine preferences by data mining from
        physicochemical properties*, Decision Support Systems, 2009.

    Note:
        The licenses of the datasets may differ from TorchUncertainty's
        license. Check before use.
    """

    md5_zip = "0ddfa7a9379510fe7ff88b9930e3c332"
    url = "https://archive.ics.uci.edu/static/public/186/wine+quality.zip"
    dataset_name = "wine_quality"
    num_features = 11

    def __init__(
        self,
        root: Path | str,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        binary: bool = True,
        download: bool = False,
        train: bool = True,
        test_split: float = 0.2,
        split_seed: int = 21893027,
        variant: str = "red",
        threshold: int = 6,
    ) -> None:
        """Wine Quality classification dataset.

        Args:
            root (str | Path): Root directory of the datasets.
            transform (callable, optional): Transform applied to each sample.
                Defaults to ``None``.
            target_transform (callable, optional): Transform applied to each
                target. Defaults to ``None``.
            binary (bool, optional): If ``True``, binarises quality scores
                using ``threshold`` (score ≥ threshold → 1). If ``False``,
                keeps raw integer quality scores. Defaults to ``True``.
            download (bool, optional): If ``True``, downloads the dataset.
                Defaults to ``False``.
            train (bool, optional): If ``True``, use the training split.
                Defaults to ``True``.
            test_split (float, optional): Fraction of data held out as test
                set. Defaults to ``0.2``.
            split_seed (int, optional): Seed for the train/test split.
                Defaults to ``21893027``.
            variant (str, optional): ``"red"`` or ``"white"``. Defaults to
                ``"red"``.
            threshold (int, optional): Quality threshold for binary mode.
                Samples with quality ≥ threshold are labelled 1. Defaults to
                ``6``.
        """
        if variant not in ("red", "white"):
            raise ValueError(f"variant must be 'red' or 'white', got {variant!r}.")
        self._variant = variant
        self._threshold = threshold
        self.filename = f"winequality-{variant}.csv"
        super().__init__(
            root=root,
            transform=transform,
            target_transform=target_transform,
            binary=binary,
            download=download,
            train=train,
            test_split=test_split,
            split_seed=split_seed,
        )

    def _check_integrity(self) -> bool:
        return (self.root / self.dataset_name / self.filename).is_file()

    def _make_dataset(self) -> None:
        data = pd.read_csv(
            self.root / self.dataset_name / self.filename,
            sep=";",
        )
        self.targets = torch.tensor(data["quality"].to_numpy().copy(), dtype=torch.long)
        data = data.drop(columns=["quality"])
        self.data = torch.tensor(data.to_numpy().copy(), dtype=torch.float32)
        self.num_features = self.data.shape[1]

    def _postprocess_targets(self, binary: bool) -> None:
        if binary:
            self.targets = (self.targets >= self._threshold).long()
