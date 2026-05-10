from torch_uncertainty.datasets.classification.tabular import GermanCredit

from .tabular_classification import TabularClassificationDataModule


class GermanCreditDataModule(TabularClassificationDataModule):
    """Datamodule for the UCI Statlog German Credit dataset."""

    dataset_class = GermanCredit
