from pathlib import Path

from torch_uncertainty.datasets.classification.tabular import WineQuality
from torch_uncertainty.datasets.utils import create_train_val_split

from .tabular_classification import TabularClassificationDataModule


class WineQualityDataModule(TabularClassificationDataModule):
    """Datamodule for the UCI Wine Quality dataset."""

    dataset_class = WineQuality

    def __init__(
        self,
        root: str | Path,
        batch_size: int,
        eval_batch_size: int | None = None,
        val_split: float = 0.0,
        test_split: float = 0.2,
        num_workers: int = 1,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        binary: bool = True,
        variant: str = "red",
        threshold: int = 6,
    ) -> None:
        """Wine Quality datamodule.

        Args:
            root (str | Path): Root directory of the datasets.
            batch_size (int): Number of samples per training batch.
            eval_batch_size (int | None): Samples per evaluation batch.
                Defaults to :attr:`batch_size`.
            val_split (float, optional): Share of training samples used for
                validation. Defaults to ``0``.
            test_split (float, optional): Share of the full dataset held out
                as test set. Defaults to ``0.2``.
            num_workers (int, optional): Data-loading subprocesses. Defaults
                to ``1``.
            pin_memory (bool, optional): Whether to pin memory. Defaults to
                ``True``.
            persistent_workers (bool, optional): Whether to keep workers alive
                between epochs. Defaults to ``True``.
            binary (bool, optional): If ``True``, binarises quality scores.
                Defaults to ``True``.
            variant (str, optional): ``"red"`` or ``"white"``. Defaults to
                ``"red"``.
            threshold (int, optional): Quality threshold for binary mode.
                Defaults to ``6``.
        """
        super().__init__(
            root=root,
            batch_size=batch_size,
            eval_batch_size=eval_batch_size,
            val_split=val_split,
            test_split=test_split,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            binary=binary,
        )
        self.variant = variant
        self.threshold = threshold

    def prepare_data(self) -> None:
        self.dataset_class(root=self.root, download=True, variant=self.variant)

    def setup(self, stage: str | None = None) -> None:
        kwargs = {
            "download": False,
            "binary": self.binary,
            "test_split": self.test_split,
            "variant": self.variant,
            "threshold": self.threshold,
        }
        if stage == "fit" or stage is None:
            full = self.dataset_class(self.root, train=True, **kwargs)
            if self.val_split:
                self.train, self.val = create_train_val_split(full, self.val_split)
            else:
                self.train = full
                self.val = self.dataset_class(self.root, train=False, **kwargs)
        if stage == "test" or stage is None:
            self.test = self.dataset_class(self.root, train=False, **kwargs)
        if stage not in ("fit", "test", None):
            raise ValueError(f"Stage {stage!r} is not supported.")
