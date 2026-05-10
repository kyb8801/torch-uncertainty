from torch_uncertainty.datasets.classification.tabular import PimaDiabetes

from .tabular_classification import TabularClassificationDataModule


class PimaDiabetesDataModule(TabularClassificationDataModule):
    """Datamodule for the UCI Pima Indians Diabetes dataset."""

    dataset_class = PimaDiabetes
