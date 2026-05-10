from torch_uncertainty.datasets.classification.tabular import TelcoChurn

from .tabular_classification import TabularClassificationDataModule


class TelcoChurnDataModule(TabularClassificationDataModule):
    """Datamodule for the Telecom Customer Churn dataset (OpenML 40701)."""

    dataset_class = TelcoChurn
