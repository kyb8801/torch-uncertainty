from torch_uncertainty.datasets.classification.tabular import DOTA2Games

from .tabular_classification import TabularClassificationDataModule


class DOTA2GamesDataModule(TabularClassificationDataModule):
    """Datamodule for the UCI DOTA 2 Games Results dataset."""

    dataset_class = DOTA2Games
