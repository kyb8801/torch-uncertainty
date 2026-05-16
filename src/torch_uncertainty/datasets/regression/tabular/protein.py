import pandas as pd
import torch

from .base import TabularRegressionDataset


class Protein(TabularRegressionDataset):
    """The UCI Physicochemical Properties of Protein Tertiary Structure dataset.

    Note:
        The licenses of the datasets may differ from TorchUncertainty's
        license. Check before use.
    """

    url = (
        "https://archive.ics.uci.edu/static/public/265/"
        "physicochemical+properties+of+protein+tertiary+structure.zip"
    )
    filename = "CASP.csv"
    dataset_name = "protein"
    md5 = "37bcb77a8abad274a987439e6a3de632"

    def _make_dataset(self) -> None:
        array = pd.read_csv(self._data_path / self.filename).to_numpy()
        self.data = torch.tensor(array[:, :-1], dtype=torch.float32)
        self.targets = torch.tensor(array[:, -1], dtype=torch.float32)
