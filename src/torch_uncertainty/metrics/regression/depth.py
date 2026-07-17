from typing import Any, Literal

import torch
from torch import Tensor
from torchmetrics import MeanAbsoluteError, MeanSquaredError, Metric
from torchmetrics.utilities.data import dim_zero_cat


class SILog(Metric):
    log_dists: Tensor
    sq_log_dists: Tensor
    total: Tensor

    def __init__(self, sqrt: bool = False, lmbda: float = 1.0, **kwargs: Any) -> None:
        r"""Compute The Scale-Invariant Logarithmic Loss metric.

        The Scale-Invariant Logarithmic Loss (SILog), a metric designed for depth estimation tasks.

        .. math:: \text{SILog} = \frac{1}{N} \sum_{i=1}^{N} \left(\log(y_i) - \log(\hat{y_i})\right)^2 - \left(\frac{1}{N} \sum_{i=1}^{N} \log(y_i) \right)^2,

        where :math:`N` is the batch size, :math:`y_i` is a tensor of target
        values and :math:`\hat{y_i}` is a tensor of prediction.
        Return the square root of SILog by setting :attr:`sqrt` to `True`.

        This metric evaluates the scale-invariant error between predicted and target values in log-space. It accounts for
        both the variance of the error and the mean log difference between predictions and targets. By setting the
        :attr:`sqrt` argument to `True`, the metric computes the square root of the SILog value.

        Args:
            sqrt: If `True`, return the square root of the metric. Defaults to ``False.``
            lmbda: The regularization parameter on the variance of error. Defaults to ``1.0.``
            kwargs: Additional keyword arguments, see `Advanced metric settings <https://torchmetrics.readthedocs.io/en/stable/pages/overview.html#metric-kwargs>`_.

        Reference:
            [1] `Depth Map Prediction from a Single Image using a Multi-Scale Deep Network, NeurIPS 2014
            <https://papers.nips.cc/paper_files/paper/2014/hash/7bccfde7714a1ebadf06c5f4cea752c1-Abstract.html>`_.

            [2] `From big to small: Multi-scale local planar guidance for monocular depth estimation
            <https://arxiv.org/abs/1907.10326>`_.

        Example:

        .. code-block:: python

            from torch_uncertainty.metrics.regression import SILog
            import torch

            # Initialize the SILog metric with sqrt=True
            silog_metric = SILog(sqrt=True, lmbda=1.0)

            # Example predictions and targets
            preds = torch.tensor([1.5, 2.0, 3.5, 5.0])
            target = torch.tensor([1.4, 2.2, 3.3, 5.2])

            # Update the metric state
            silog_metric.update(preds, target)

            # Compute the Scale-Invariant Logarithmic Loss
            result = silog_metric.compute()
            print(f"SILog: {result.item():.4f}")
            # Output: SILog: 0.0686

        """
        super().__init__(**kwargs)
        self.sqrt = sqrt
        self.lmbda = lmbda
        self.add_state(
            "log_dists",
            default=torch.tensor(0.0),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "sq_log_dists",
            default=torch.tensor(0.0),
            dist_reduce_fx="sum",
        )
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:  # pyrefly: ignore[bad-override]
        """Update state with predictions and targets.

        Args:
            preds: A prediction tensor of shape (batch)
            target: A tensor of ground truth labels of shape (batch)
        """
        self.log_dists += torch.sum(preds.log() - target.log())
        self.sq_log_dists += torch.sum((preds.log() - target.log()) ** 2)
        self.total += target.size(0)

    def compute(self) -> Tensor:
        """Compute the Scale-Invariant Logarithmic Loss."""
        log_dists = dim_zero_cat(self.log_dists)
        sq_log_dists = dim_zero_cat(self.sq_log_dists)
        out = sq_log_dists / self.total - self.lmbda * log_dists**2 / (self.total * self.total)
        if self.sqrt:
            return torch.sqrt(out)
        return out


class MeanGTRelativeAbsoluteError(MeanAbsoluteError):
    def __init__(self, **kwargs) -> None:
        r"""Compute the Mean Absolute Error relative to the Ground Truth (MAErel or ARErel).

        This metric is commonly used in tasks where the relative deviation of
        predictions with respect to the ground truth is important.

        .. math:: \text{MAErel} = \frac{1}{N}\sum_i^N \frac{| y_i - \hat{y_i} |}{y_i}

        where :math:`y` is a tensor of target values, and :math:`\hat{y}` is a
        tensor of predictions.

        As input to ``forward`` and ``update`` the metric accepts the following
        input:

        - **preds** (:class:`~torch.Tensor`): Predictions from model
        - **target** (:class:`~torch.Tensor`): Ground truth values

        As output of ``forward`` and ``compute`` the metric returns the
        following output:

        - **rel_mean_absolute_error** (:class:`~torch.Tensor`): A tensor with
          the relative mean absolute error over the state

        Args:
            kwargs: Additional keyword arguments, see `Advanced metric settings <https://torchmetrics.readthedocs.io/en/stable/pages/overview.html#metric-kwargs>`_.

        Reference:
            [1] `From big to small: Multi-scale local planar guidance for monocular depth estimation
            <https://arxiv.org/abs/1907.10326>`_.

        Example:

        .. code-block:: python

            from torch_uncertainty.metrics.regression import MeanGTRelativeAbsoluteError
            import torch

            # Initialize the metric
            mae_rel_metric = MeanGTRelativeAbsoluteError()

            # Example predictions and targets
            preds = torch.tensor([2.5, 1.0, 2.0, 8.0])
            target = torch.tensor([3.0, 1.5, 2.0, 7.0])

            # Update the metric state
            mae_rel_metric.update(preds, target)

            # Compute the Relative Mean Absolute Error
            result = mae_rel_metric.compute()
            print(f"Relative Mean Absolute Error: {result.item()}")
            # Output: 0.1607142984867096

        .. seealso::
            - :class:`MeanGTRelativeSquaredError`
        """
        super().__init__(**kwargs)

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        return super().update(preds / target, torch.ones_like(target))


class MeanGTRelativeSquaredError(MeanSquaredError):
    def __init__(self, squared: bool = True, num_outputs: int = 1, **kwargs) -> None:
        r"""Compute mean squared error relative to the Ground Truth (MSErel or SRE).

        This metric is useful for evaluating the relative squared error between
        predictions and targets, particularly in regression tasks where relative
        accuracy is critical.

        .. math:: \text{MSErel} = \frac{1}{N}\sum_i^N \frac{(y_i - \hat{y_i})^2}{y_i}

        Where :math:`y` is a tensor of target values, and :math:`\hat{y}` is a
        tensor of predictions.

        As input to ``forward`` and ``update`` the metric accepts the following
        input:

        - **preds** (:class:`~torch.Tensor`): Predictions from model
        - **target** (:class:`~torch.Tensor`): Ground truth values

        As output of ``forward`` and ``compute`` the metric returns the
        following output:

        - **rel_mean_squared_error** (:class:`~torch.Tensor`): A tensor with
          the relative mean squared error

        Args:
            squared: If True returns MSErel value, if False returns RMSErel value.
            num_outputs: Number of outputs in multioutput setting
            kwargs: Additional keyword arguments, see `Advanced metric settings
              <https://torchmetrics.readthedocs.io/en/stable/pages/overview.html#metric-kwargs>`_.

        Reference:
            [1] `From big to small: Multi-scale local planar guidance for monocular depth estimation
            <https://arxiv.org/abs/1907.10326>`_.

        Example:

        .. code-block:: python

            from torch_uncertainty.metrics.regression import MeanGTRelativeSquaredError
            import torch

            # Initialize the metric
            mse_rel_metric = MeanGTRelativeSquaredError(squared=True)

            # Example predictions and targets
            preds = torch.tensor([2.5, 1.0, 2.0, 8.0])
            target = torch.tensor([3.0, 1.5, 2.0, 7.0])

            # Update the metric state
            mse_rel_metric.update(preds, target)

            # Compute the Relative Mean Squared Error
            result = mse_rel_metric.compute()
            print(f"Relative Mean Squared Error: {result.item()}")
            # Output: 0.09821434319019318

        .. seealso::
            - :class:`MeanGTRelativeAbsoluteError`
        """
        super().__init__(squared, num_outputs, **kwargs)

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        return super().update(preds / torch.sqrt(target), torch.sqrt(target))


class MeanSquaredLogError(MeanSquaredError):
    def __init__(self, squared: bool = True, **kwargs) -> None:
        r"""Compute the Mean Squared Logarithmic Error (MSLE).

        This metric is commonly used in regression problems where the relative
        difference between predictions and targets is of greater importance than
        the absolute difference. It is particularly effective for datasets with
        wide-ranging magnitudes, as it penalizes underestimation more than
        overestimation.

        .. math:: \text{MSELog} = \frac{1}{N}\sum_i^N  (\log \hat{y_i} - \log y_i)^2

        where  :math:`y`  is a tensor of target values, and :math:`\hat{y}` is a
        tensor of predictions.

        As input to ``forward`` and ``update`` the metric accepts the following
        input:

        - **preds** (:class:`~torch.Tensor`): Predictions from model
        - **target** (:class:`~torch.Tensor`): Ground truth values

        As output of ``forward`` and ``compute`` the metric returns the
        following output:

            - **mse_log** (:class:`~torch.Tensor`): A tensor with the
                mean squared logarithmic error over the state

        Args:
            squared: If ``True``, returns MSLE. If ``False``, returns RMSLE.
            kwargs: Additional keyword arguments, see `Advanced metric settings <https://torchmetrics.readthedocs.io/en/stable/pages/overview.html#metric-kwargs>`_.

        Reference:
            [1] `From big to small: Multi-scale local planar guidance for monocular depth estimation
            <https://arxiv.org/abs/1907.10326>`_.

        Example:

        .. code-block:: python

            from torch_uncertainty.metrics.regression import MeanSquaredLogError
            import torch

            # Initialize the metric
            msle_metric = MeanSquaredLogError(squared=True)

            # Example predictions and targets (must be non-negative)
            preds = torch.tensor([2.5, 1.0, 2.0, 8.0])
            target = torch.tensor([3.0, 1.5, 2.0, 7.0])

            # Update the metric state
            msle_metric.update(preds, target)

            # Compute the Mean Squared Logarithmic Error
            result = msle_metric.compute()
            print(f"Mean Squared Logarithmic Error: {result.item()}")
            # Output: Mean Squared Logarithmic Error: 0.05386843904852867
        """
        super().__init__(squared, **kwargs)

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        return super().update(preds.log(), target.log())


class Log10(MeanAbsoluteError):
    def __init__(self, **kwargs) -> None:
        r"""Compute the LOG10 metric.

        The Log10 metric computes the mean absolute error in the base-10 logarithmic space.

        .. math:: \text{Log10} = \frac{1}{N} \sum_{i=1}^{N} |\log_{10}(y_i) - \log_{10}(\hat{y_i})|

        where:
        - :math:`N` is the number of elements in the batch.
        - :math:`y_i` represents the true target values.
        - :math:`\hat{y_i}` represents the predicted values.

        This metric is useful when data spans multiple orders of magnitude,
        where evaluating error in log-space provides a more meaningful
        comparison.

        Inputs:
            - :attr:`preds`: :math:`(N)`
            - :attr:`target`: :math:`(N)`

        Args:
            kwargs: Additional keyword arguments, see `Advanced metric settings <https://torchmetrics.readthedocs.io/en/stable/pages/overview.html#metric-kwargs>`_.

        Example:

        .. code-block:: python

            from torch_uncertainty.metrics.regression import Log10
            import torch

            # Initialize the metric
            log10_metric = Log10()

            # Example predictions and targets
            preds = torch.tensor([10.0, 100.0, 1000.0])
            target = torch.tensor([12.0, 95.0, 1020.0])

            # Update the metric state
            log10_metric.update(preds, target)

            # Compute the Log10 error
            result = log10_metric.compute()
            print(f"Log10 Error: {result.item()}")
            # Output: Log10 Error: 0.03668594
        """
        super().__init__(**kwargs)
        self.add_state("values", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        return super().update(preds.log10(), target.log10())


def _unit_to_factor(unit: Literal["mm", "m", "km"]) -> float:
    """Convert a unit to a factor for scaling.

    Args:
        unit: Unit for the computation of the metric. Must be one of 'mm', 'm',
            'km'.
    """
    if unit == "km":
        return 1e-3
    if unit == "m":
        return 1.0
    if unit == "mm":
        return 1e3
    raise ValueError(f"unit must be one of 'mm', 'm', 'km'. Got {unit}.")


class MeanSquaredErrorInverse(MeanSquaredError):
    def __init__(
        self,
        squared: bool = True,
        num_outputs: int = 1,
        unit: Literal["mm", "m", "km"] = "km",
        **kwargs,
    ) -> None:
        r"""Mean Squared Error of the inverse predictions (iMSE).

        .. math:: \text{iMSE} = \frac{1}{N}\sum_i^N(\frac{1}{y_i} - \frac{1}{\hat{y_i}})^2

        Where :math:`y` is a tensor of target values, and :math:`\hat{y}` is a
        tensor of predictions.
        Both are scaled by a factor of :attr:`unit_factor` depending on the
        :attr:`unit` given.

        As input to ``forward`` and ``update`` the metric accepts the following
        input:

        - **preds** (:class:`~Tensor`): Predictions from model
        - **target** (:class:`~Tensor`): Ground truth values

        As output of ``forward`` and ``compute`` the metric returns the following
        output:

        - **mean_squared_error** (:class:`~Tensor`): A tensor with the mean
          squared error

        Args:
            squared: If ``True``, returns MSE. If ``False``, returns RMSE.
            num_outputs: Number of outputs in multioutput setting.
            unit: Unit for the computation of the metric. Must be one of
                ``"mm"``, ``"m"``, ``"km"``. Defaults to ``"km"``.
            kwargs: Additional keyword arguments.
        """
        super().__init__(squared, num_outputs, **kwargs)
        self.unit_factor = _unit_to_factor(unit)

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        super().update(1 / (preds * self.unit_factor), 1 / (target * self.unit_factor))


class MeanAbsoluteErrorInverse(MeanAbsoluteError):
    def __init__(self, unit: Literal["mm", "m", "km"] = "km", **kwargs) -> None:
        r"""Mean Absolute Error of the inverse predictions (iMAE).

        .. math:: \text{iMAE} = \frac{1}{N}\sum_i^N \left| \frac{1}{y_i} - \frac{1}{\hat{y_i}} \right|

        Where :math:`y` is a tensor of target values, and :math:`\hat{y}` is a
        tensor of predictions.
        Both are scaled by a factor of :attr:`unit_factor` depending on the
        :attr:`unit` given.

        As input to ``forward`` and ``update`` the metric accepts the following
        input:

        - **preds** (:class:`~Tensor`): Predictions from model
        - **target** (:class:`~Tensor`): Ground truth values

        As output of ``forward`` and ``compute`` the metric returns the following
        output:

        - **mean_absolute_inverse_error** (:class:`~Tensor`): A tensor with the
          mean absolute error over the state

        Args:
            unit: Unit for the computation of the metric. Must be one of
                ``"mm"``, ``"m"``, ``"km"``. Defaults to ``"km"``.
            kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.unit_factor = _unit_to_factor(unit)

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        super().update(1 / (preds * self.unit_factor), 1 / (target * self.unit_factor))


class ThresholdAccuracy(Metric):
    values: Tensor
    total: Tensor

    def __init__(self, power: int, lmbda: float = 1.25, **kwargs) -> None:
        r"""Compute the Threshold Accuracy metric, also referred to as :math:`\delta_1`,
        :math:`\delta_2`, or :math:`\delta_3`.

        This metric is standard in monocular depth estimation. It reports the fraction
        of pixels (or samples) whose prediction :math:`\hat{y}_i` and target :math:`y_i`
        agree up to a multiplicative factor :math:`\lambda^k`:

        .. math::
            \delta_k = \frac{1}{N} \sum_{i=1}^{N}
            \mathbf{1}\!\left[ \max\!\left(\frac{\hat{y}_i}{y_i},
            \frac{y_i}{\hat{y}_i}\right) < \lambda^k \right],

        where :math:`\lambda = 1.25` by default and :math:`k` is the :attr:`power`
        argument. Higher values are better.

        Args:
            power: The power to raise the threshold to. Often in [1, 2, 3].
            lmbda: The threshold to compare the max of ratio of predictions
                to targets and its inverse to. Defaults to ``1.25.``
            kwargs: Additional keyword arguments, see `Advanced metric settings
                <https://torchmetrics.readthedocs.io/en/stable/pages/overview.html#metric-kwargs>`_.

        Example:

        .. code-block:: python

            from torch_uncertainty.metrics.regression import ThresholdAccuracy
            import torch

            # Initialize the metric with power=2 and lambda=1.25
            threshold_accuracy = ThresholdAccuracy(power=2, lmbda=1.25)

            # Example predictions and targets
            preds = torch.tensor([2.0, 3.0, 5.0, 8.0, 20.0])
            target = torch.tensor([2.1, 2.5, 4.5, 10.0, 10.0])

            # Update the metric state
            threshold_accuracy.update(preds, target)

            # Compute the Threshold Accuracy
            result = threshold_accuracy.compute()
            print(f"Threshold Accuracy: {result.item():.2f}")
            # Output: Threshold Accuracy: 0.80
        """
        super().__init__(**kwargs)
        if power < 0:
            raise ValueError(f"Power must be greater than or equal to 0. Got {power}.")
        self.power = power
        if lmbda < 1:
            raise ValueError(f"Lambda must be greater than or equal to 1. Got {lmbda}.")
        self.lmbda = lmbda
        self.add_state("values", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:  # pyrefly: ignore[bad-override]
        """Update state with predictions and targets."""
        self.values += torch.sum(torch.max(preds / target, target / preds) < self.lmbda**self.power)
        self.total += target.size(0)

    def compute(self) -> Tensor:
        """Compute the Threshold Accuracy."""
        values = dim_zero_cat(self.values)
        return values / self.total
