import warnings
from urllib.error import URLError

import pytest

from tests._dummies.dataset import DummyRegressionDataset
from torch_uncertainty.datamodules import (
    Kin8NMDataModule,
    TabularRegressionDataModule,
    WineQualityRegressionDataModule,
)


class TestTabularRegressionDataModuleAPI:
    """Test the TabularRegressionDataModule base-class contract."""

    def test_missing_dataset_class_raises(self) -> None:
        with pytest.raises(TypeError):
            TabularRegressionDataModule(root="./data/", batch_size=128)

    def test_invalid_stage_raises(self) -> None:
        dm = Kin8NMDataModule(root="./data/", batch_size=128)
        dm.dataset_class = DummyRegressionDataset
        dm.prepare_data()
        dm.setup()
        with pytest.raises(ValueError):
            dm.setup("other")


class TestKin8NMDataModule:
    """Testing Kin8NMDataModule as a representative regression datamodule."""

    def test_kin8nm(self) -> None:
        dm = Kin8NMDataModule(root="./data/", batch_size=128)
        dm.dataset_class = DummyRegressionDataset
        dm.prepare_data()
        dm.setup()

        dm.train_dataloader()
        dm.val_dataloader()
        dm.test_dataloader()

        dm.setup("fit")
        dm.setup("test")

    def test_kin8nm_val_split(self) -> None:
        dm = Kin8NMDataModule(root="./data/", batch_size=128, val_split=0.5)
        dm.dataset_class = DummyRegressionDataset
        dm.prepare_data()
        dm.setup()

        dm.train_dataloader()
        dm.val_dataloader()
        dm.test_dataloader()


class TestWineQualityRegressionDataModule:
    """Testing WineQualityRegressionDataModule (variant param)."""

    def test_wine_quality_regression(self) -> None:
        try:
            dm = WineQualityRegressionDataModule(root="./data/", batch_size=128, variant="red")
            dm.prepare_data()
            dm.setup()

            dm.train_dataloader()
            dm.val_dataloader()
            dm.test_dataloader()

            dm = WineQualityRegressionDataModule(
                root="./data/", batch_size=128, variant="white", val_split=0.1
            )
            dm.prepare_data()
            dm.setup()
        except URLError as e:
            warnings.warn(f"Data download failed due to network error: {e}", stacklevel=2)
