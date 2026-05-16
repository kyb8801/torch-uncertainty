import warnings
from urllib.error import URLError

import pytest

from tests._dummies.dataset import DummyRegressionDataset
from torch_uncertainty.datamodules.classification import (
    AdultCensusIncomeDataModule,
    AmazonAccessDataModule,
    APSFailureDataModule,
    BankMarketingDataModule,
    CreditApprovalDataModule,
    DOTA2GamesDataModule,
    GermanCreditDataModule,
    HiggsBosonDataModule,
    HTRU2DataModule,
    KDDChurnDataModule,
    OnlineShoppersDataModule,
    PimaDiabetesDataModule,
    SpamBaseDataModule,
    TabularClassificationDataModule,
    TelcoChurnDataModule,
    WineQualityDataModule,
)


class TestTabularDataModuleAPI:
    """Test the TabularClassificationDataModule base-class contract."""

    def test_missing_dataset_class_raises(self) -> None:
        with pytest.raises(TypeError):
            TabularClassificationDataModule(root="./data/", batch_size=128)

    def test_invalid_stage_raises(self) -> None:
        dm = HTRU2DataModule(root="./data/", batch_size=128)
        dm.dataset_class = DummyRegressionDataset
        dm.prepare_data()
        dm.setup()
        with pytest.raises(ValueError):
            dm.setup("other")


class TestHTRU2DataModule:
    """Testing HTRU2DataModule as a representative classification datamodule."""

    def test_htru2(self) -> None:
        dm = HTRU2DataModule(root="./data/", batch_size=128)
        dm.dataset_class = DummyRegressionDataset
        dm.prepare_data()
        dm.setup()

        dm.train_dataloader()
        dm.val_dataloader()
        dm.test_dataloader()

        dm.setup("fit")
        dm.setup("test")

    def test_htru2_val_split(self) -> None:
        dm = HTRU2DataModule(root="./data/", batch_size=128, val_split=0.5)
        dm.dataset_class = DummyRegressionDataset
        dm.prepare_data()
        dm.setup()

        dm.train_dataloader()
        dm.val_dataloader()
        dm.test_dataloader()


class TestWineQualityDataModule:
    """Testing WineQualityDataModule (variant parameter)."""

    def test_wine_quality_classification(self) -> None:
        try:
            dm = WineQualityDataModule(root="./data/", batch_size=128, variant="red")
            dm.prepare_data()
            dm.setup()

            dm.train_dataloader()
            dm.val_dataloader()
            dm.test_dataloader()

            dm = WineQualityDataModule(
                root="./data/", batch_size=128, variant="white", val_split=0.1
            )
            dm.prepare_data()
            dm.setup()
        except URLError as e:
            warnings.warn(f"Data download failed due to network error: {e}", stacklevel=2)


class TestOtherClassificationDataModules:
    """Smoke-test instantiation of all remaining classification datamodules."""

    def test_all_modules_instantiate(self) -> None:
        AdultCensusIncomeDataModule(root="./data/", batch_size=128)
        AmazonAccessDataModule(root="./data/", batch_size=128)
        APSFailureDataModule(root="./data/", batch_size=128)
        BankMarketingDataModule(root="./data/", batch_size=128)
        CreditApprovalDataModule(root="./data/", batch_size=128)
        DOTA2GamesDataModule(root="./data/", batch_size=128)
        GermanCreditDataModule(root="./data/", batch_size=128)
        HiggsBosonDataModule(root="./data/", batch_size=128)
        KDDChurnDataModule(root="./data/", batch_size=128)
        OnlineShoppersDataModule(root="./data/", batch_size=128)
        PimaDiabetesDataModule(root="./data/", batch_size=128)
        SpamBaseDataModule(root="./data/", batch_size=128)
        TelcoChurnDataModule(root="./data/", batch_size=128)
