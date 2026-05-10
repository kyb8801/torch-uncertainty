import pandas as pd
import torch
from torchvision.datasets.utils import download_url

from .tabular_classification import TabularClassificationDataset, _load_arff


class TelcoChurn(TabularClassificationDataset):
    """The Telecom Customer Churn dataset (OpenML 40701).

    Predicts whether a customer churns. The ``phone_number`` column is dropped
    as it is a non-predictive identifier. Downloaded from the OpenML repository
    as an ARFF file.

    Note:
        The licenses of the datasets may differ from TorchUncertainty's
        license. Check before use.
    """

    # OpenML dataset 40701, file_id 4965302
    url = "https://api.openml.org/data/v1/download/4965302"
    dataset_name = "telco_churn"
    filename = "churn.arff"
    is_archive = False

    def download(self) -> None:
        if self._check_integrity():
            return
        (self.root / self.dataset_name).mkdir(parents=True, exist_ok=True)
        download_url(self.url, root=str(self.root / self.dataset_name), filename=self.filename)

    def _make_dataset(self) -> None:
        df = _load_arff(self.root / self.dataset_name / self.filename)
        # Drop non-predictive identifier
        df = df.drop(columns=["phone_number"], errors="ignore")
        target_col = "class"
        target_vals = df[target_col].str.strip().str.lower()
        self.targets = torch.as_tensor((target_vals == "true").astype(int).values, dtype=torch.long)
        df = df.drop(columns=[target_col])
        cat_cols = df.select_dtypes(include="object").columns
        df = pd.get_dummies(df, columns=cat_cols).astype(float)
        self.data = torch.as_tensor(df.values, dtype=torch.float32)
        self.num_features = self.data.shape[1]
