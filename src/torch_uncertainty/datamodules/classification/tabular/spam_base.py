from torch_uncertainty.datasets.classification.tabular import SpamBase

from .tabular_classification import TabularClassificationDataModule


class SpamBaseDataModule(TabularClassificationDataModule):
    """Datamodule for the UCI SpamBase e-mail spam dataset."""

    dataset_class = SpamBase
