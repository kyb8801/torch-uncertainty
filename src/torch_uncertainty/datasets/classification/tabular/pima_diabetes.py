import pandas as pd
import torch

from .tabular_classification import TabularClassificationDataset

_PIMA_COLUMNS = [
    "pregnancies",
    "glucose",
    "blood_pressure",
    "skin_thickness",
    "insulin",
    "bmi",
    "diabetes_pedigree",
    "age",
    "outcome",
]


class PimaDiabetes(TabularClassificationDataset):
    """The UCI Pima Indians Diabetes dataset.

    Predicts diabetes onset from clinical measurements. All features are
    numeric. The dataset is downloaded from the UCI ML Repository.

    Reference:
        J.W. Smith et al., *Using the ADAP Learning Algorithm to Forecast the
        Onset of Diabetes Mellitus*, Proc. SCAMC, 1988.

    Note:
        The licenses of the datasets may differ from TorchUncertainty's
        license. Check before use.
    """

    url = "https://archive.ics.uci.edu/static/public/34/diabetes.zip"
    dataset_name = "pima_diabetes"
    filename = "pima-indians-diabetes.data"

    def _make_dataset(self) -> None:
        data = pd.read_csv(
            self.root / self.dataset_name / self.filename,
            names=_PIMA_COLUMNS,
        )
        self.targets = torch.as_tensor(data["outcome"].values, dtype=torch.long)
        data = data.drop(columns=["outcome"])
        self.data = torch.as_tensor(data.values.astype(float), dtype=torch.float32)
        self.num_features = self.data.shape[1]
