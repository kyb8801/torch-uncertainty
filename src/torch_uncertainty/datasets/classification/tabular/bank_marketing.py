import logging

import numpy as np
import pandas as pd
import torch
from torchvision.datasets.utils import download_and_extract_archive, extract_archive

from .base import TabularClassificationDataset


class BankMarketing(TabularClassificationDataset):
    """The UCI Bank Marketing dataset.

    Predicts whether a client subscribes to a term deposit from direct
    marketing campaign data.

    Reference:
        S. Moro et al., *A Data-Driven Approach to Predict the Success of Bank
        Telemarketing*, Decision Support Systems, 2014.

    Note:
        The licenses of the datasets may differ from TorchUncertainty's
        license. Check before use.
    """

    md5_zip = "3a3c6c4189975ea1f3040dbd60ad106c"
    url = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
    dataset_name = "bank_marketing"
    filename = "bank-additional-full.csv"

    def _check_integrity(self) -> bool:
        return (self.root / self.dataset_name / "bank-additional" / self.filename).is_file()

    def download(self) -> None:
        if self._check_integrity():
            logging.info("Files already downloaded and verified")
            return
        download_and_extract_archive(
            self.url,
            download_root=self.root / self.dataset_name,
            filename="bank+marketing.zip",
            md5=self.md5_zip,
        )
        extract_archive(
            self.root / self.dataset_name / "bank-additional.zip",
            self.root / self.dataset_name,
        )

    def _make_dataset(self) -> None:
        data = pd.read_csv(
            self.root / self.dataset_name / "bank-additional" / self.filename,
            sep=";",
        )
        self.targets = torch.as_tensor(np.where(data["y"] == "yes", 1, 0), dtype=torch.long)
        data = data.drop(columns=["y"])
        binary_cols = data.select_dtypes(include="object").columns
        for col in binary_cols:
            if data[col].nunique() == 2:
                data[col] = np.where(data[col] == "yes", 1, 0)
        self.data = torch.as_tensor(pd.get_dummies(data).astype(float).values, dtype=torch.float32)
        self.num_features = self.data.shape[1]
