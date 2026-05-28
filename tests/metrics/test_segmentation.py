import pytest
import torch

from torch_uncertainty.metrics.segmentation import (
    MeanIntersectionOverUnion,
    PAvPU,
    SegmentationMetric,
)
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


class TestPAvPU:
    def test_update_and_compute(self) -> None:
        metric = PAvPU(patch_size=2, acc_threshold=0.5, unc_threshold=0.4 + 1e-6)
        # Test the example from the paper (https://arxiv.org/pdf/1811.12709) showcased in Figure 2.
        target = torch.tensor([[[1, 2, 5, 7], [6, 4, 3, 3], [10, 9, 5, 0], [8, 6, 4, 4]]])
        preds = torch.tensor(
            [
                [
                    [
                        [0.01, 0.9, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
                        [0.03, 0.03, 0.7, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03],
                        [0.06, 0.06, 0.06, 0.06, 0.4, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06],
                        [0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.7, 0.03, 0.03, 0.03],
                    ],
                    [
                        [0.07, 0.07, 0.07, 0.07, 0.07, 0.3, 0.07, 0.07, 0.07, 0.07, 0.07],
                        [0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.4, 0.06, 0.06, 0.06, 0.06],
                        [0.02, 0.02, 0.02, 0.8, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02],
                        [0.01, 0.01, 0.01, 0.9, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
                    ],
                    [
                        [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.8],
                        [0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.6, 0.04],
                        [0.05, 0.05, 0.05, 0.05, 0.5, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
                        [0.7, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03],
                    ],
                    [
                        [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.9, 0.01, 0.01],
                        [0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.3, 0.07, 0.07, 0.07],
                        [0.06, 0.06, 0.06, 0.4, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06],
                        [0.02, 0.02, 0.02, 0.02, 0.8, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02],
                    ],
                ]
            ]
        ).permute(0, 3, 1, 2)
        metric.update(preds, target)
        result = metric.compute()
        assert result == torch.tensor(0.75)


class TestSegmentationMetric:
    def _inner(self, num_classes: int = 3) -> MeanIntersectionOverUnion:
        return MeanIntersectionOverUnion(num_classes=num_classes)

    # --- constructor ---

    def test_invalid_subsampling_rate_zero(self) -> None:
        with pytest.raises(ValueError, match="subsampling_rate"):
            SegmentationMetric(self._inner(), subsampling_rate=0.0)

    def test_invalid_subsampling_rate_negative(self) -> None:
        with pytest.raises(ValueError, match="subsampling_rate"):
            SegmentationMetric(self._inner(), subsampling_rate=-0.5)

    def test_invalid_subsampling_rate_above_one(self) -> None:
        with pytest.raises(ValueError, match="subsampling_rate"):
            SegmentationMetric(self._inner(), subsampling_rate=1.1)

    def test_valid_subsampling_rate_boundary(self) -> None:
        metric = SegmentationMetric(self._inner(), subsampling_rate=1.0)
        assert metric.subsampling_rate == 1.0

    def test_valid_subsampling_rate_none(self) -> None:
        metric = SegmentationMetric(self._inner(), subsampling_rate=None)
        assert metric.subsampling_rate is None

    # --- update: target shapes ---

    def test_update_target_3d(self) -> None:
        metric = SegmentationMetric(self._inner())
        preds = torch.rand(2, 3, 4, 4)
        target = torch.randint(0, 3, (2, 4, 4))
        metric.update(preds, target)
        result = metric.compute()
        assert result.ndim == 0

    def test_update_target_4d_single_channel(self) -> None:
        metric = SegmentationMetric(self._inner())
        preds = torch.rand(2, 3, 4, 4)
        target = torch.randint(0, 3, (2, 1, 4, 4))
        metric.update(preds, target)
        result = metric.compute()
        assert result.ndim == 0

    def test_update_target_4d_multichannel_raises(self) -> None:
        metric = SegmentationMetric(self._inner())
        preds = torch.rand(2, 3, 4, 4)
        target = torch.randint(0, 3, (2, 3, 4, 4))
        with pytest.raises(ValueError, match="Expected target"):
            metric.update(preds, target)

    # --- update: ignore_mask shapes ---

    def test_update_with_ignore_mask_3d(self) -> None:
        metric = SegmentationMetric(self._inner())
        preds = torch.rand(2, 3, 4, 4)
        target = torch.randint(0, 3, (2, 4, 4))
        ignore_mask = torch.zeros(2, 4, 4, dtype=torch.bool)
        ignore_mask[:, :2, :] = True
        metric.update(preds, target, ignore_mask=ignore_mask)
        result = metric.compute()
        assert result.ndim == 0

    def test_update_with_ignore_mask_4d_single_channel(self) -> None:
        metric = SegmentationMetric(self._inner())
        preds = torch.rand(2, 3, 4, 4)
        target = torch.randint(0, 3, (2, 4, 4))
        ignore_mask = torch.zeros(2, 1, 4, 4, dtype=torch.bool)
        metric.update(preds, target, ignore_mask=ignore_mask)
        result = metric.compute()
        assert result.ndim == 0

    def test_update_with_ignore_mask_4d_multichannel_raises(self) -> None:
        metric = SegmentationMetric(self._inner())
        preds = torch.rand(2, 3, 4, 4)
        target = torch.randint(0, 3, (2, 4, 4))
        ignore_mask = torch.zeros(2, 3, 4, 4, dtype=torch.bool)
        with pytest.raises(ValueError, match="Expected ignore_mask"):
            metric.update(preds, target, ignore_mask=ignore_mask)

    # --- subsampling ---

    def test_update_with_subsampling(self) -> None:
        metric = SegmentationMetric(self._inner(), subsampling_rate=0.5)
        preds = torch.rand(2, 3, 8, 8)
        target = torch.randint(0, 3, (2, 8, 8))
        metric.update(preds, target)
        result = metric.compute()
        assert result.ndim == 0

    def test_update_with_subsampling_and_ignore_mask(self) -> None:
        metric = SegmentationMetric(self._inner(), subsampling_rate=0.5)
        preds = torch.rand(2, 3, 8, 8)
        target = torch.randint(0, 3, (2, 8, 8))
        ignore_mask = torch.zeros(2, 8, 8, dtype=torch.bool)
        ignore_mask[:, :4, :] = True
        metric.update(preds, target, ignore_mask=ignore_mask)
        result = metric.compute()
        assert result.ndim == 0

    # --- compute / reset delegation ---

    def test_compute_delegates_to_inner_metric(self) -> None:
        inner = self._inner()
        metric = SegmentationMetric(inner)
        preds = torch.rand(2, 3, 4, 4)
        target = torch.randint(0, 3, (2, 4, 4))
        metric.update(preds, target)
        assert torch.equal(metric.compute(), inner.compute())

    def test_reset_clears_inner_metric(self) -> None:
        metric = SegmentationMetric(self._inner())
        preds = torch.rand(2, 3, 4, 4)
        target = torch.randint(0, 3, (2, 4, 4))
        metric.update(preds, target)
        result_before = metric.compute()
        metric.reset()
        metric.update(preds, target)
        result_after = metric.compute()
        assert torch.isclose(result_before, result_after)

    def test_multiple_updates(self) -> None:
        metric = SegmentationMetric(self._inner())
        for _ in range(3):
            preds = torch.rand(2, 3, 4, 4)
            target = torch.randint(0, 3, (2, 4, 4))
            metric.update(preds, target)
        result = metric.compute()
        assert 0.0 <= result.item() <= 1.0
