from torch_uncertainty.datasets.classification.tabular import OnlineShoppers

from .tabular_classification import TabularClassificationDataModule


class OnlineShoppersDataModule(TabularClassificationDataModule):
    """Datamodule for the UCI Online Shoppers Purchasing Intention dataset."""

    dataset_class = OnlineShoppers
