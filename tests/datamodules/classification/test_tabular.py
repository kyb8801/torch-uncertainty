import warnings
from urllib.error import URLError

import pytest

from torch_uncertainty.datamodules.classification import (
    BankMarketingDataModule,
    DOTA2GamesDataModule,
    HTRU2DataModule,
    OnlineShoppersDataModule,
    SpamBaseDataModule,
    TabularClassificationDataModule,
)


class TestTabularDataModuleAPI:
    """Test the TabularClassificationDataModule base-class contract."""

    def test_missing_dataset_class_raises(self) -> None:
        with pytest.raises(TypeError):
            TabularClassificationDataModule(root="./data/", batch_size=128)

    def test_invalid_stage_raises(self) -> None:
        try:
            dm = HTRU2DataModule(root="./data/", batch_size=128)
            dm.prepare_data()
            dm.setup()
            with pytest.raises(ValueError):
                dm.setup("other")
        except URLError as e:
            warnings.warn(f"Data download failed due to network error: {e}", stacklevel=2)


class TestHTRU2DataModule:
    """Testing the HTRU2DataModule datamodule class."""

    def test_htru2(self) -> None:
        try:
            dm = HTRU2DataModule(root="./data/", batch_size=128)
            dm.prepare_data()
            dm.setup()

            dm.train_dataloader()
            dm.val_dataloader()
            dm.test_dataloader()

            dm.setup("test")
            dm.test_dataloader()

            dm = HTRU2DataModule(root="./data/", batch_size=128, val_split=0.1)
            dm.prepare_data()
            dm.setup()
        except URLError as e:
            warnings.warn(f"Data download failed due to network error: {e}", stacklevel=2)

    def test_other_modules_instantiate(self) -> None:
        try:
            BankMarketingDataModule(root="./data/", batch_size=128)
            DOTA2GamesDataModule(root="./data/", batch_size=128)
            OnlineShoppersDataModule(root="./data/", batch_size=128)
            SpamBaseDataModule(root="./data/", batch_size=128)
        except URLError as e:
            warnings.warn(f"Data download failed due to network error: {e}", stacklevel=2)
