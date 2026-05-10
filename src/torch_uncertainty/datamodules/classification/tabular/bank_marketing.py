from torch_uncertainty.datasets.classification.tabular import BankMarketing

from .tabular_classification import TabularClassificationDataModule


class BankMarketingDataModule(TabularClassificationDataModule):
    """Datamodule for the UCI Bank Marketing dataset."""

    dataset_class = BankMarketing
