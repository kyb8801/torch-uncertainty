from torch_uncertainty.datasets.classification.tabular import HiggsBoson

from .tabular_classification import TabularClassificationDataModule


class HiggsBosonDataModule(TabularClassificationDataModule):
    """Datamodule for the Higgs Boson dataset (OpenML 23512)."""

    dataset_class = HiggsBoson
