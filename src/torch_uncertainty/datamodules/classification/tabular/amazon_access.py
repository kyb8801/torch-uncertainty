from torch_uncertainty.datasets.classification.tabular import AmazonAccess

from .tabular_classification import TabularClassificationDataModule


class AmazonAccessDataModule(TabularClassificationDataModule):
    """Datamodule for the Amazon Employee Access dataset."""

    dataset_class = AmazonAccess
