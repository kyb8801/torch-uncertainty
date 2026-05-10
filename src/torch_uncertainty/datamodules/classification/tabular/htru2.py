from torch_uncertainty.datasets.classification.tabular import HTRU2

from .tabular_classification import TabularClassificationDataModule


class HTRU2DataModule(TabularClassificationDataModule):
    """Datamodule for the UCI HTRU2 pulsar dataset."""

    dataset_class = HTRU2
