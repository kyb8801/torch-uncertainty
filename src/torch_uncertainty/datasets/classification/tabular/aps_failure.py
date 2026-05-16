import logging

import pandas as pd
import torch
from torchvision.datasets.utils import download_and_extract_archive

from .tabular_classification import TabularClassificationDataset


class APSFailure(TabularClassificationDataset):
    """The UCI APS Failure at Scania Trucks dataset.

    Predicts whether an air pressure system (APS) component caused a truck
    failure. The dataset is provided pre-split; ``train=False`` loads the
    held-out test set. Missing values (``na``) are imputed with the column mean.

    Reference:
        M. Cerqueira et al., *Predicting Failures in Industrial Plants*,
        UCI ML Repository, 2016.

    Note:
        The licenses of the datasets may differ from TorchUncertainty's
        license. Check before use.
    """

    url = "https://archive.ics.uci.edu/static/public/421/aps+failure+at+scania+trucks.zip"
    dataset_name = "aps_failure"
    filename = "aps_failure_training_set.csv"
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
            filename="aps_failure.zip",
        )

    def _make_dataset(self) -> None:
        fname = "aps_failure_training_set.csv" if self.train else "aps_failure_test_set.csv"
        # The CSV files begin with ~20 comment lines followed by the header row.
        data = pd.read_csv(
            self.root / self.dataset_name / fname,
            na_values=["na"],
            comment=None,
            header=0,
            skiprows=20,
        )
        self.targets = torch.tensor(
            (data["class"] == "pos").astype(int).to_numpy(), dtype=torch.long
        )
        data = data.drop(columns=["class"])
        # Impute missing values with column mean
        data = data.apply(pd.to_numeric, errors="coerce")
        data = data.fillna(data.mean())
        self.data = torch.tensor(data.to_numpy(dtype=float), dtype=torch.float32)
        self.num_features = self.data.shape[1]
