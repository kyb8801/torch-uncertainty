import pandas as pd
import torch

from .tabular_classification import TabularClassificationDataset, _load_arff


class AmazonAccess(TabularClassificationDataset):
    """The Amazon Employee Access dataset (OpenML 41135).

    Predicts whether an employee's request for access to a resource should be
    granted, based on role and resource identifiers. All nine predictive features
    are categorical (encoded as integers). Downloaded from OpenML as an ARFF.

    Note:
        The exact OpenML dataset identifier for this dataset should be verified
        before use. The URL below targets OpenML dataset 41135; if the download
        fails, check ``https://www.openml.org`` for the correct file identifier.

        The licenses of the datasets may differ from TorchUncertainty's
        license. Check before use.
    """

    # OpenML dataset ~41135 — verify the file_id at openml.org if download fails
    url = "https://api.openml.org/data/v1/download/21820967"
    dataset_name = "amazon_access"
    filename = "amazon_employee_access.arff"
    is_archive = False

    def download(self) -> None:
        if self._check_integrity():
            return
        from torchvision.datasets.utils import download_url

        (self.root / self.dataset_name).mkdir(parents=True, exist_ok=True)
        download_url(self.url, root=str(self.root / self.dataset_name), filename=self.filename)

    def _make_dataset(self) -> None:
        df = _load_arff(self.root / self.dataset_name / self.filename)
        target_col = "target" if "target" in df.columns else df.columns[-1]
        self.targets = torch.as_tensor(df[target_col].astype(int).values, dtype=torch.long)
        df = df.drop(columns=[target_col])
        cat_cols = df.select_dtypes(include="object").columns
        df = pd.get_dummies(df, columns=cat_cols).astype(float)
        # Treat all columns as categorical (high-cardinality integer IDs)
        self.data = torch.as_tensor(df.values, dtype=torch.float32)
        self.num_features = self.data.shape[1]
