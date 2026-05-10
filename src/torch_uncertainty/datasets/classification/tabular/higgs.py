import torch
from torchvision.datasets.utils import download_url

from .tabular_classification import TabularClassificationDataset, _load_arff


class HiggsBoson(TabularClassificationDataset):
    """The Higgs Boson dataset — small version (OpenML 23512, 98 050 samples).

    Predicts whether a collision event produces a Higgs boson or is background
    noise. All features are numerical. Downloaded from the OpenML repository as
    an ARFF file.

    Reference:
        Baldi et al., *Searching for Exotic Particles in High-Energy Physics with
        Deep Learning*, Nature Communications, 2014.

    Note:
        The licenses of the datasets may differ from TorchUncertainty's
        license. Check before use.
    """

    # OpenML dataset 23512, file_id 2063675
    url = "https://api.openml.org/data/v1/download/2063675"
    dataset_name = "higgs_boson"
    filename = "higgs.arff"
    is_archive = False

    def download(self) -> None:
        if self._check_integrity():
            return
        (self.root / self.dataset_name).mkdir(parents=True, exist_ok=True)
        download_url(self.url, root=str(self.root / self.dataset_name), filename=self.filename)

    def _make_dataset(self) -> None:
        df = _load_arff(self.root / self.dataset_name / self.filename)
        target_col = "class"
        self.targets = torch.as_tensor(
            df[target_col].astype(float).astype(int).values, dtype=torch.long
        )
        df = df.drop(columns=[target_col])
        self.data = torch.as_tensor(df.values.astype(float), dtype=torch.float32)
        self.num_features = self.data.shape[1]
