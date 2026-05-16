import pandas as pd
import torch

from .base import TabularRegressionDataset


class EnergyEfficiency(TabularRegressionDataset):
    """The UCI Energy Efficiency dataset.

    Predicts the heating load of buildings. The first two feature columns and
    the last three columns of the raw Excel file are dropped (they are either
    uninformative or the second target).

    Note:
        The licenses of the datasets may differ from TorchUncertainty's
        license. Check before use.
    """

    url = "https://archive.ics.uci.edu/static/public/242/energy+efficiency.zip"
    filename = "ENB2012_data.xlsx"
    dataset_name = "energy-efficiency"
    md5 = "2018fb7b50778fdc1304d50a78874579"

    def _make_dataset(self) -> None:
        array = pd.read_excel(self._data_path / self.filename).to_numpy()
        self.data = torch.tensor(array[:, 2:-3], dtype=torch.float32)
        self.targets = torch.tensor(array[:, -2], dtype=torch.float32)
