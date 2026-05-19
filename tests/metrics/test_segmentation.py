import torch

from torch_uncertainty.metrics.segmentation.seg_binary_auroc import SegmentationBinaryAUROC
from torch_uncertainty.metrics.segmentation.seg_binary_average_precision import (
    SegmentationBinaryAveragePrecision,
)
from torch_uncertainty.metrics.segmentation.seg_fpr95 import SegmentationFPR95


class TestSegmentationBinaryAUROC:
    def test_update_and_compute(self) -> None:
        metric = SegmentationBinaryAUROC()
        preds = torch.rand(4, 64)
        target = torch.randint(0, 2, (4, 64))
        metric.update(preds, target)
        result = metric.compute()
        assert result.ndim == 0

    def test_compute_zero_total(self) -> None:
        metric = SegmentationBinaryAUROC()
        result = metric.compute()
        assert result == 0.0

    def test_multiple_batches(self) -> None:
        metric = SegmentationBinaryAUROC()
        for _ in range(3):
            preds = torch.rand(2, 32)
            target = torch.randint(0, 2, (2, 32))
            metric.update(preds, target)
        result = metric.compute()
        assert 0.0 <= result.item() <= 1.0


class TestSegmentationBinaryAveragePrecision:
    def test_update_and_compute(self) -> None:
        metric = SegmentationBinaryAveragePrecision()
        preds = torch.rand(4, 64)
        target = torch.randint(0, 2, (4, 64))
        metric.update(preds, target)
        result = metric.compute()
        assert result.ndim == 0

    def test_compute_zero_total(self) -> None:
        metric = SegmentationBinaryAveragePrecision()
        result = metric.compute()
        assert result == 0.0

    def test_multiple_batches(self) -> None:
        metric = SegmentationBinaryAveragePrecision()
        for _ in range(3):
            preds = torch.rand(2, 32)
            target = torch.randint(0, 2, (2, 32))
            metric.update(preds, target)
        result = metric.compute()
        assert 0.0 <= result.item() <= 1.0


class TestSegmentationFPR95:
    def test_update_and_compute(self) -> None:
        metric = SegmentationFPR95(pos_label=1)
        # 1D tensors: N pixels per image, passed one image at a time
        preds = torch.cat([torch.ones(50) * 0.9, torch.ones(50) * 0.1])
        target = torch.cat([torch.ones(50, dtype=torch.long), torch.zeros(50, dtype=torch.long)])
        metric.update(preds, target)
        result = metric.compute()
        assert result.ndim == 0

    def test_compute_zero_total(self) -> None:
        metric = SegmentationFPR95(pos_label=1)
        result = metric.compute()
        assert torch.isnan(result)

    def test_multiple_batches(self) -> None:
        metric = SegmentationFPR95(pos_label=1)
        for _ in range(3):
            preds = torch.cat([torch.ones(50) * 0.9, torch.ones(50) * 0.1])
            target = torch.cat([torch.ones(50), torch.zeros(50)]).long()
            metric.update(preds, target)
        result = metric.compute()
        assert result.ndim == 0
