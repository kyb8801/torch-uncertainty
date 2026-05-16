import pandas as pd
import torch

from .base import TabularRegressionDataset


class NavalPropulsionPlant(TabularRegressionDataset):
    """The UCI Condition Based Maintenance of Naval Propulsion Plants dataset.

    Predicts the gas turbine compressor decay state coefficient. The second
    target column is dropped.

    Note:
        The licenses of the datasets may differ from TorchUncertainty's
        license. Check before use.
    """

    url = "https://raw.githubusercontent.com/luishpinto/cm-naval-propulsion-plant/master/data.csv"
    filename = "data.csv"
    dataset_name = "naval-propulsion-plant"
    is_archive = False
    md5 = "54f4febcf51bdba12e1ca63e28b3e973"

    def _make_dataset(self) -> None:
        df = pd.read_csv(
            self._data_path / self.filename,
            header=None,
            sep=";",
            decimal=",",
        )
        array = df.apply(pd.to_numeric, errors="coerce").to_numpy()
        self.data = torch.tensor(array[:, :-2], dtype=torch.float32)
        self.targets = torch.tensor(array[:, -2], dtype=torch.float32)
