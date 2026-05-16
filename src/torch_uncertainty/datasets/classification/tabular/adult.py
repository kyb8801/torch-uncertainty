import logging

import numpy as np
import pandas as pd
import torch
from torchvision.datasets.utils import download_and_extract_archive

from .base import TabularClassificationDataset

_ADULT_COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]


class AdultCensusIncome(TabularClassificationDataset):
    """The UCI Adult Census Income dataset.

    Predicts whether income exceeds $50K/year. The ``fnlwgt`` sampling-weight
    column is dropped as it is not a predictive feature.

    Reference:
        R. Kohavi, *Scaling Up the Accuracy of Naive-Bayes Classifiers: a
        Decision-Tree Hybrid*, KDD, 1996.

    Note:
        The licenses of the datasets may differ from TorchUncertainty's
        license. Check before use.
    """

    url = "https://archive.ics.uci.edu/static/public/2/adult.zip"
    dataset_name = "adult"
    filename = "adult.data"
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
            filename="adult.zip",
        )

    def _make_dataset(self) -> None:
        fname = "adult.data" if self.train else "adult.test"
        skiprows = 1 if not self.train else 0  # test file has a header comment
        data = pd.read_csv(
            self.root / self.dataset_name / fname,
            names=_ADULT_COLUMNS,
            skipinitialspace=True,
            na_values=["?"],
            skiprows=skiprows,
        )
        # Strip trailing period from the test file target column
        data["income"] = data["income"].str.strip().str.rstrip(".")
        self.targets = torch.as_tensor(np.where(data["income"] == ">50K", 1, 0), dtype=torch.long)
        # Drop target and the non-predictive sampling weight
        data = data.drop(columns=["income", "fnlwgt"])
        # Fill missing categorical values with mode
        for col in data.select_dtypes(include="object").columns:
            data[col] = data[col].fillna(data[col].mode()[0])
        self.data = torch.as_tensor(pd.get_dummies(data).astype(float).values, dtype=torch.float32)
        self.num_features = self.data.shape[1]
