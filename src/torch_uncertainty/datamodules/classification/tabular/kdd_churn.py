from torch_uncertainty.datasets.classification.tabular import KDDChurn

from .tabular_classification import TabularClassificationDataModule


class KDDChurnDataModule(TabularClassificationDataModule):
    """Datamodule for the KDD Cup 2009 Customer Churn dataset (OpenML 1112)."""

    dataset_class = KDDChurn
