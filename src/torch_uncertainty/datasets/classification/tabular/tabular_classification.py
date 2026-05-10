import gzip
import io
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import torch
from torch import Generator
from torch.utils.data import Dataset
from torchvision.datasets.utils import (
    download_and_extract_archive,
    download_url,
)


def _load_arff(path: Path) -> pd.DataFrame:
    """Parse an ARFF file into a pandas DataFrame.

    Handles both plain text and gzip-compressed ARFF files.
    """
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            content = f.read()
    except (gzip.BadGzipFile, OSError):
        with path.open(encoding="utf-8") as f:
            content = f.read()

    col_names = []
    data_start = 0
    lines = content.splitlines()

    for i, line in enumerate(lines):
        stripped = line.strip()
        lower = stripped.lower()
        if not stripped or stripped.startswith("%"):
            continue
        if lower.startswith("@relation"):
            continue
        if lower.startswith("@attribute"):
            parts = stripped.split(None, 2)
            col_names.append(parts[1].strip("'\""))
        elif lower == "@data":
            data_start = i + 1
            break

    data_content = "\n".join(lines[data_start:])
    return pd.read_csv(
        io.StringIO(data_content),
        header=None,
        names=col_names,
        na_values=["?", ""],
        skipinitialspace=True,
        quotechar="'",
    )


class TabularClassificationDataset(Dataset, ABC):
    """Base class for tabular binary classification datasets.

    Subclasses must define the class attributes :attr:`url`, :attr:`filename`,
    :attr:`dataset_name` and implement :meth:`_make_dataset`.

    If the source is a plain file rather than a zip archive, set
    :attr:`is_archive` to ``False``; :meth:`download` will then call
    :func:`~torchvision.datasets.utils.download_url` instead of
    :func:`~torchvision.datasets.utils.download_and_extract_archive`.
    """

    md5_zip: str | None = None
    url: str = ""
    filename: str = ""
    dataset_name: str = ""
    num_features: int = 0
    need_split: bool = True
    apply_standardization: bool = True
    is_archive: bool = True

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
    ) -> None:
        """Tabular binary classification dataset.

        Args:
            root (str | Path): Root directory of the datasets.
            transform (callable, optional): A function/transform that takes in a
                tensor and returns a transformed version. Defaults to ``None``.
            target_transform (callable, optional): A function/transform that takes
                in the target and transforms it. Defaults to ``None``.
            binary (bool, optional): If ``True``, returns scalar targets; otherwise
                one-hot encodes them into two classes. Defaults to ``True``.
            download (bool, optional): If ``True``, downloads the dataset from the
                internet. If already present, it is not downloaded again. Defaults
                to ``False``.
            train (bool, optional): If ``True``, use the training split. Defaults
                to ``True``.
            test_split (float, optional): Fraction of the dataset held out as test
                set when :attr:`need_split` is ``True``. Defaults to ``0.2``.
            split_seed (int, optional): Random seed for the train/test split.
                Defaults to ``21893027``.

        Note:
            The licenses of the datasets may differ from TorchUncertainty's
            license. Check before use.
        """
        super().__init__()
        self.root = Path(root)
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        if download:
            self.download()

        self._make_dataset()
        if self.apply_standardization:
            self._compute_statistics()
            self._standardize()

        if self.need_split:
            gen = Generator().manual_seed(split_seed)
            self.split_idx = torch.ones(len(self)).multinomial(
                num_samples=int((1 - test_split) * len(self)),
                replacement=False,
                generator=gen,
            )
            if not self.train:
                self.split_idx = torch.tensor(
                    [i for i in range(len(self)) if i not in self.split_idx]
                )
            self.data = self.data[self.split_idx]
            self.targets = self.targets[self.split_idx]
        self._postprocess_targets(binary)

    def __len__(self) -> int:
        """Get the number of rows of the tabular data."""
        return self.data.shape[0]

    def _check_integrity(self) -> bool:
        return (self.root / self.dataset_name / self.filename).is_file()

    def _standardize(self) -> None:
        self.data = (self.data - self.data_mean) / self.data_std

    def _compute_statistics(self) -> None:
        self.data_mean = self.data.mean(dim=0)
        self.data_std = self.data.std(dim=0)
        self.data_std[self.data_std == 0] = 1

    def download(self) -> None:
        """Download and, if needed, extract the dataset."""
        if self._check_integrity():
            logging.info("Files already downloaded and verified")
            return
        download_root = self.root / self.dataset_name
        if self.is_archive:
            download_and_extract_archive(
                self.url,
                download_root=download_root,
                filename=self.dataset_name + ".zip",
                md5=self.md5_zip,
            )
        else:
            download_url(
                self.url,
                root=str(download_root),
                filename=self.filename,
                md5=self.md5_zip,
            )

    def _postprocess_targets(self, binary: bool) -> None:
        """Post-process targets after splitting.

        The default behaviour one-hot encodes targets into two classes when
        ``binary`` is ``False``. Override this in subclasses that require
        different target handling (e.g. multi-class datasets).
        """
        if not binary:
            self.targets = torch.nn.functional.one_hot(self.targets, num_classes=2)

    @abstractmethod
    def _make_dataset(self) -> None:
        """Populate ``self.data`` (float32 tensor) and ``self.targets`` (long tensor)."""

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the row of id index of the tabular data."""
        data = self.data[index, :]
        if self.transform is not None:
            data = self.transform(data)
        target = self.targets[index]
        if self.target_transform is not None:
            target = self.target_transform(target)
        return data, target
