import pandas as pd
import torch

from .base import TabularClassificationDataset


class GermanCredit(TabularClassificationDataset):
    """The UCI Statlog German Credit dataset.

    Predicts credit risk (good/bad). All features use the integer-coded
    representation from ``german.data``; categorical attributes are encoded
    with ``pd.get_dummies``.

    Reference:
        H. Hofmann, *Statlog (German Credit Data)*, UCI ML Repository, 1994.

    Note:
        The licenses of the datasets may differ from TorchUncertainty's
        license. Check before use.
    """

    url = "https://archive.ics.uci.edu/static/public/144/statlog+german+credit+data.zip"
    dataset_name = "german_credit"
    filename = "german.data"

    def _make_dataset(self) -> None:
        data = pd.read_csv(
            self.root / self.dataset_name / self.filename,
            sep=r"\s+",
            header=None,
        )
        # Last column: 1 = good credit → 0, 2 = bad credit → 1
        self.targets = torch.as_tensor((data.iloc[:, -1].values - 1), dtype=torch.long)
        data = data.iloc[:, :-1]
        self.data = torch.as_tensor(pd.get_dummies(data).astype(float).values, dtype=torch.float32)
        self.num_features = self.data.shape[1]
