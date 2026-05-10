from torch_uncertainty.datasets.classification.tabular import APSFailure

from .tabular_classification import TabularClassificationDataModule


class APSFailureDataModule(TabularClassificationDataModule):
    """Datamodule for the UCI APS Failure at Scania Trucks dataset."""

    dataset_class = APSFailure
