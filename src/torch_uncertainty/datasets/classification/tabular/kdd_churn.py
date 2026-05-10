import pandas as pd
import torch
from torchvision.datasets.utils import download_url

from .tabular_classification import TabularClassificationDataset, _load_arff


class KDDChurn(TabularClassificationDataset):
    """The KDD Cup 2009 Customer Churn dataset (OpenML 1112).

    Predicts telecommunications customer churn from 190 numeric and 40
    categorical features. Missing values are imputed with the column mean
    (numeric) or the mode (categorical). Downloaded from OpenML as an ARFF.

    Reference:
        G. Lemaitre et al., *Challenges in Representation Learning: A Report
        on Three Machine Learning Contests*, KDD Cup, 2009.

    Note:
        The licenses of the datasets may differ from TorchUncertainty's
        license. Check before use.
    """

    # OpenML dataset 1112, file_id 53995
    url = "https://api.openml.org/data/v1/download/53995"
    dataset_name = "kdd_churn"
    filename = "KDDCup09_churn.arff"
    is_archive = False

    def download(self) -> None:
        if self._check_integrity():
            return
        (self.root / self.dataset_name).mkdir(parents=True, exist_ok=True)
        download_url(self.url, root=str(self.root / self.dataset_name), filename=self.filename)

    def _make_dataset(self) -> None:
        df = _load_arff(self.root / self.dataset_name / self.filename)
        target_col = "CHURN"
        self.targets = torch.as_tensor(
            (df[target_col].astype(float) > 0).astype(int).values, dtype=torch.long
        )
        df = df.drop(columns=[target_col])
        # Impute missing values
        for col in df.columns:
            if df[col].dtype == object:
                mode = df[col].mode()
                df[col] = df[col].fillna(mode[0] if len(mode) else "unknown")
            else:
                df[col] = df[col].fillna(df[col].mean())
        cat_cols = df.select_dtypes(include="object").columns
        df = pd.get_dummies(df, columns=cat_cols).astype(float)
        self.data = torch.as_tensor(df.values, dtype=torch.float32)
        self.num_features = self.data.shape[1]
