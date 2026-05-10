from torch_uncertainty.datasets.classification.tabular import CreditApproval

from .tabular_classification import TabularClassificationDataModule


class CreditApprovalDataModule(TabularClassificationDataModule):
    """Datamodule for the UCI Credit Approval dataset."""

    dataset_class = CreditApproval
