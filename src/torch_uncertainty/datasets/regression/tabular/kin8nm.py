import pandas as pd
import torch

from .base import TabularRegressionDataset


class Kin8NM(TabularRegressionDataset):
    """The Kin8NM robot arm kinematics dataset.

    Note:
        The licenses of the datasets may differ from TorchUncertainty's
        license. Check before use.
    """

    url = "https://huggingface.co/datasets/torch-uncertainty/kin8nm/raw/main/kin8nm.csv"
    filename = "kin8nm.csv"
    dataset_name = "kin8nm"
    is_archive = False
    md5 = "df08c665b7665809e74e32b107836a3a"

    def _make_dataset(self) -> None:
        array = pd.read_csv(self._data_path / self.filename).to_numpy()
        self.data = torch.tensor(array[:, :-1], dtype=torch.float32)
        self.targets = torch.tensor(array[:, -1], dtype=torch.float32)
