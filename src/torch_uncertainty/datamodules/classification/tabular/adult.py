from torch_uncertainty.datasets.classification.tabular import AdultCensusIncome

from .tabular_classification import TabularClassificationDataModule


class AdultCensusIncomeDataModule(TabularClassificationDataModule):
    """Datamodule for the UCI Adult Census Income dataset."""

    dataset_class = AdultCensusIncome
