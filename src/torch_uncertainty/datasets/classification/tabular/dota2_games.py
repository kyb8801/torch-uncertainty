import logging

import numpy as np
import pandas as pd
import torch
from torchvision.datasets.utils import download_and_extract_archive

from .base import TabularClassificationDataset


class DOTA2Games(TabularClassificationDataset):
    """The UCI DOTA 2 Games Results dataset.

    Predicts the winning team from hero selection in DOTA 2 matches.

    Note:
        The licenses of the datasets may differ from TorchUncertainty's
        license. Check before use.
    """

    md5_zip = "896623c082b062f56b9c49c6c1fc0bf7"
    url = "https://archive.ics.uci.edu/static/public/367/dota2+games+results.zip"
    dataset_name = "dota2_games"
    filename = "dota2Train.csv"
    num_features = 116
    need_split = False

    def _check_integrity(self) -> bool:
        return (self.root / self.dataset_name / self.filename).is_file()

    def download(self) -> None:
        if self._check_integrity():
            logging.info("Files already downloaded and verified")
            return
        download_and_extract_archive(
            self.url,
            download_root=self.root / self.dataset_name,
            filename="dota2+games+results.zip",
            md5=self.md5_zip,
        )

    def _make_dataset(self) -> None:
        fname = "dota2Train.csv" if self.train else "dota2Test.csv"
        data = pd.read_csv(self.root / self.dataset_name / fname, header=None)
        self.targets = torch.as_tensor(np.where(data.iloc[:, 0] == 1, 1, 0), dtype=torch.long)
        data = data.drop(columns=[0])
        self.data = torch.as_tensor(pd.get_dummies(data).astype(float).values, dtype=torch.float32)
        self.num_features = self.data.shape[1]
