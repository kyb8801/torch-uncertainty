import torch
from torch.utils.data import Dataset


class AggregatedDataset(Dataset):
    def __init__(self, dataset: Dataset, num_dataloaders: int) -> None:
        """A class to help aggregating datasets.

        Virtually interlace multiple copies of a dataset to train ensembles with
        different batch orders.

        Args:
            dataset: The dataset to be interlaced.
            num_dataloaders: The number of dataloaders to be used for training.
        """
        super().__init__()
        self.dataset = dataset
        self.num_dataloaders = num_dataloaders
        self.dataset_size = len(dataset)
        self.offset = self.dataset_size // self.num_dataloaders

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the samples and targets of the dataset.

        Args:
            idx: The index of the sample.
        """
        inputs, targets = zip(
            *[
                self.dataset[(idx + i * self.offset) % self.dataset_size]
                for i in range(self.num_dataloaders)
            ],
            strict=True,
        )
        inputs = torch.cat(inputs, dim=0)
        targets = torch.as_tensor(targets)
        return inputs, targets

    def __len__(self) -> int:
        """The number of samples in the dataset."""
        return self.dataset_size
