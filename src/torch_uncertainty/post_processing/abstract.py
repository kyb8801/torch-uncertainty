from abc import ABC, abstractmethod

from torch import Tensor, nn
from torch.utils.data import DataLoader


class PostProcessing(nn.Module, ABC):
    def __init__(self, model: nn.Module | None = None) -> None:
        """Base class for post-processing modules."""
        super().__init__()
        self.model = model
        self.trained = False

    def set_model(self, model: nn.Module) -> None:
        """Attach a model to the post-processing module."""
        self.model = model

    @abstractmethod
    def fit(self, dataloader: DataLoader) -> None:
        """Fit the post-processing module on a calibration dataloader."""

    @abstractmethod
    def forward(self, inputs: Tensor) -> Tensor:
        """Transform model outputs into post-processed predictions."""
