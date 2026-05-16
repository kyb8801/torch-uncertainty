import pandas as pd
import torch

from .base import TabularRegressionDataset


class Concrete(TabularRegressionDataset):
    """The UCI Concrete Compressive Strength dataset.

    Note:
        The licenses of the datasets may differ from TorchUncertainty's
        license. Check before use.
    """

    url = "https://archive.ics.uci.edu/static/public/165/concrete+compressive+strength.zip"
    filename = "Concrete_Data.xls"
    dataset_name = "concrete"
    md5 = "eba3e28907d4515244165b6b2c311b7b"

    def _make_dataset(self) -> None:
        array = pd.read_excel(self._data_path / self.filename).to_numpy()
        self.data = torch.tensor(array[:, :-1], dtype=torch.float32)
        self.targets = torch.tensor(array[:, -1], dtype=torch.float32)
