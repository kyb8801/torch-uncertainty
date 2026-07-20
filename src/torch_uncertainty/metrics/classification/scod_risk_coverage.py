import torch
from torch import Tensor

from .risk_coverage import AUGRC, AURC, CovAtxRisk, RiskAtxCov


class _SCODRiskCoverageMixin:
    scores: list[Tensor]
    errors: list[Tensor]

    def _set_ood_cost(self, ood_cost: float) -> None:
        if not isinstance(ood_cost, float):
            raise TypeError(f"Expected ood_cost to be of type float, but got {type(ood_cost)}")
        if not 0 <= ood_cost <= 1:
            raise ValueError(f"ood_cost should be in the range [0, 1], but got {ood_cost}.")
        self.ood_cost = ood_cost

    def update(  # pyrefly: ignore[bad-override]
        self,
        ood_scores: Tensor,
        classification_errors: Tensor,
        is_ood: Tensor,
    ) -> None:
        """Store acceptance scores and their associated SCOD losses.

        Args:
            ood_scores: Scores for which larger values indicate more OOD-like inputs.
            classification_errors: Boolean indicators of misclassified ID inputs.
                Values corresponding to OOD inputs are ignored.
            is_ood: Boolean indicators of OOD inputs.
        """
        ood_scores = ood_scores.reshape(-1)
        classification_errors = classification_errors.reshape(-1)
        is_ood = is_ood.reshape(-1)

        if not (ood_scores.numel() == classification_errors.numel() == is_ood.numel()):
            raise ValueError(
                "ood_scores, classification_errors and is_ood must contain "
                "the same number of elements."
            )

        classification_errors = classification_errors.to(dtype=ood_scores.dtype)
        is_ood = is_ood.bool()

        scod_losses = torch.where(
            is_ood,
            torch.full_like(classification_errors, self.ood_cost),
            classification_errors * (1 - self.ood_cost),
        )

        # Risk-coverage metrics rank larger scores as more likely to be accepted.
        self.scores.append(-ood_scores)
        self.errors.append(scod_losses)


class SCODAURC(_SCODRiskCoverageMixin, AURC):
    r"""Calculate the Area Under the SCOD Risk-Coverage curve.

    Selective Classification with Out-of-Distribution Detection (SCOD)
    evaluates a classifier and a rejection rule jointly. Unlike ordinary OOD
    detection, SCOD penalizes both accepted OOD samples and accepted,
    misclassified in-distribution (ID) samples.

    Let :math:`s_i` be an OOD score for sample :math:`i`, where larger values
    indicate that a sample is more likely to be OOD. Let
    :math:`o_i \in \{0, 1\}` indicate whether the sample is OOD and let
    :math:`e_i \in \{0, 1\}` indicate whether an ID sample is misclassified.
    For an OOD acceptance cost :math:`c_{\mathrm{OOD}} \in [0, 1]`, the
    per-sample SCOD loss is

    .. math::

        \ell_i =
        \begin{cases}
            0,
            & o_i = 0,\ e_i = 0,\\
            1 - c_{\mathrm{OOD}},
            & o_i = 0,\ e_i = 1,\\
            c_{\mathrm{OOD}},
            & o_i = 1.
        \end{cases}

    Equivalently,

    .. math::

        \ell_i =
        c_{\mathrm{OOD}} o_i
        + (1 - c_{\mathrm{OOD}})(1-o_i)e_i.

    This convention corresponds to the cost :math:`c_{\mathrm{fn}}` used by
    Narasimhan et al. The parameter :math:`\beta` used by Xia and Bouganis
    instead denotes the ID misclassification cost, and therefore satisfies

    .. math::

        \beta = 1 - c_{\mathrm{OOD}}.

    Let :math:`\sigma` be the permutation sorting the :math:`N` samples by
    increasing OOD score,

    .. math::

        s_{\sigma(1)} \leq \cdots \leq s_{\sigma(N)},

    so that samples most likely to be accepted appear first. At empirical
    coverage :math:`\kappa_k = k/N`, the SCOD selective risk is

    .. math::

        r(\kappa_k)
        = \frac{1}{k}\sum_{j=1}^{k}\ell_{\sigma(j)}.

    The SCOD-AURC is the area under this joint risk-coverage curve:

    .. math::

        \operatorname{SCOD\text{-}AURC}
        = \int_0^1 r(\kappa)\,\mathrm{d}\kappa.

    The discrete integration and finite-sample normalization follow
    :class:`~torch_uncertainty.metrics.classification.AURC`.

    As input to ``forward`` and ``update``, the metric accepts:

    - **ood_scores** (:class:`~torch.Tensor`): Float tensor containing one OOD
      score per sample. Larger values must indicate more OOD-like samples.
    - **classification_errors** (:class:`~torch.Tensor`): Boolean or binary
      tensor indicating misclassified ID samples. Values corresponding to OOD
      samples are ignored.
    - **is_ood** (:class:`~torch.Tensor`): Boolean or binary tensor indicating
      which samples are OOD.

    As output to ``forward`` and ``compute``, the metric returns:

    - **scod_aurc** (:class:`~torch.Tensor`): Scalar tensor containing the area
      under the SCOD risk-coverage curve. Lower values are better.

    Args:
        ood_cost: Relative cost :math:`c_{\mathrm{OOD}}` of accepting an OOD
            sample. The cost of accepting a misclassified ID sample is
            ``1 - ood_cost``. Defaults to ``0.5``.
        kwargs: Additional keyword arguments passed to
            :class:`torchmetrics.Metric`.

    Note:
        The empirical ratio of ID to OOD samples determines the mixture
        proportion evaluated by this metric. Results obtained with different
        ID/OOD ratios are therefore not directly comparable unless the mixture
        proportions are controlled.

    References:
        [1] `Xia & Bouganis. Augmenting Softmax Information for Selective
        Classification with Out-of-Distribution Data. ACCV, 2022.
        <https://openaccess.thecvf.com/content/ACCV2022/papers/Xia_Augmenting_Softmax_Information_for_Selective_Classification_with_Out-of-Distribution_Data_ACCV_2022_paper.pdf>`_.

        [2] `Narasimhan et al. Plugin Estimators for Selective Classification
        with Out-of-Distribution Detection.
        <https://arxiv.org/abs/2301.12386>`_.

        [3] `Geifman & El-Yaniv. Selective Classification for Deep Neural
        Networks. NeurIPS, 2017.
        <https://papers.nips.cc/paper_files/paper/2017/file/4a8423d5e91fda00bb7e46540e2b0cf1-Paper.pdf>`_.
    """

    def __init__(self, ood_cost: float = 0.5, **kwargs) -> None:
        super().__init__(**kwargs)
        self._set_ood_cost(ood_cost)


class SCODAUGRC(_SCODRiskCoverageMixin, AUGRC):
    r"""Calculate the Area Under the Generalized SCOD Risk-Coverage curve.

    This metric applies the generalized risk-coverage construction of
    :class:`~torch_uncertainty.metrics.classification.AUGRC` to the joint SCOD
    loss defined by
    :class:`~torch_uncertainty.metrics.classification.SCODAURC`.

    Let :math:`r(\kappa)` denote the SCOD selective risk at coverage
    :math:`\kappa`. The generalized SCOD risk is
    :math:`\kappa r(\kappa)`, and its area is

    .. math::

        \operatorname{SCOD\text{-}AUGRC}
        = \int_0^1 \kappa r(\kappa)\,\mathrm{d}\kappa.

    At empirical coverage :math:`\kappa_k=k/N`, this uses

    .. math::

        \kappa_k r(\kappa_k)
        = \frac{1}{N}\sum_{j=1}^{k}\ell_{\sigma(j)},

    where :math:`\ell_i` is the SCOD loss and :math:`\sigma` orders samples
    from lowest to highest OOD score.

    As input to ``forward`` and ``update``, the metric accepts:

    - **ood_scores** (:class:`~torch.Tensor`): Float tensor containing OOD
      scores, where larger values indicate more OOD-like samples.
    - **classification_errors** (:class:`~torch.Tensor`): Boolean or binary
      tensor indicating misclassified ID samples.
    - **is_ood** (:class:`~torch.Tensor`): Boolean or binary tensor indicating
      OOD samples.

    As output to ``forward`` and ``compute``, the metric returns:

    - **scod_augrc** (:class:`~torch.Tensor`): Scalar tensor containing the area
      under the generalized SCOD risk-coverage curve. Lower values are better.

    Args:
        ood_cost: Relative cost :math:`c_{\mathrm{OOD}}` of accepting an OOD
            sample. The cost of accepting a misclassified ID sample is
            ``1 - ood_cost``. Defaults to ``0.5``.
        kwargs: Additional keyword arguments passed to
            :class:`torchmetrics.Metric`.

    Note:
        SCOD-AUGRC is the Torch-Uncertainty generalized risk-coverage metric
        applied to the SCOD loss. It is a natural SCOD extension of AUGRC, but
        it is not introduced under this name in the original SCOD papers.

    References:
        [1] `Xia & Bouganis. Augmenting Softmax Information for Selective
        Classification with Out-of-Distribution Data. ACCV, 2022.
        <https://openaccess.thecvf.com/content/ACCV2022/papers/Xia_Augmenting_Softmax_Information_for_Selective_Classification_with_Out-of-Distribution_Data_ACCV_2022_paper.pdf>`_.

        [2] `Narasimhan et al. Plugin Estimators for Selective Classification
        with Out-of-Distribution Detection.
        <https://arxiv.org/abs/2301.12386>`_.

        [3] `Traub et al. Overcoming Common Flaws in the Evaluation of
        Selective Classification Systems.
        <https://arxiv.org/abs/2407.01032>`_.

    See Also:
        :class:`~torch_uncertainty.metrics.classification.SCODAURC`:
            Definition of the SCOD loss and selective risk.
        :class:`~torch_uncertainty.metrics.classification.AUGRC`:
            Original generalized risk-coverage metric.
    """

    def __init__(self, ood_cost: float = 0.5, **kwargs) -> None:
        super().__init__(**kwargs)
        self._set_ood_cost(ood_cost)


class SCODCovAtxRisk(_SCODRiskCoverageMixin, CovAtxRisk):
    r"""Calculate the maximum coverage at a specified SCOD risk.

    This metric applies
    :class:`~torch_uncertainty.metrics.classification.CovAtxRisk` to the joint
    SCOD loss defined by
    :class:`~torch_uncertainty.metrics.classification.SCODAURC`.

    Let :math:`r(\kappa_k)` denote the empirical SCOD selective risk at
    coverage :math:`\kappa_k=k/N`. For a risk threshold
    :math:`\tau\in[0,1]`, coverage at SCOD risk :math:`\tau` is

    .. math::

        \operatorname{SCOD\text{-}Cov@Risk}(\tau)
        =
        \max\left\{
            \kappa_k :
            r(\kappa_k)\leq\tau,\quad k\in\{1,\ldots,N\}
        \right\}.

    If no positive coverage satisfies the risk constraint, the metric returns
    ``nan``. Because empirical selective risk need not be monotonic in
    coverage, the metric considers every available coverage and returns the
    largest admissible one.

    As input to ``forward`` and ``update``, the metric accepts:

    - **ood_scores** (:class:`~torch.Tensor`): Float tensor containing OOD
      scores, where larger values indicate more OOD-like samples.
    - **classification_errors** (:class:`~torch.Tensor`): Boolean or binary
      tensor indicating misclassified ID samples.
    - **is_ood** (:class:`~torch.Tensor`): Boolean or binary tensor indicating
      OOD samples.

    As output to ``forward`` and ``compute``, the metric returns:

    - **scod_coverage** (:class:`~torch.Tensor`): Scalar tensor containing the
      maximum coverage satisfying the SCOD risk constraint. Higher values are
      better.

    Args:
        risk_threshold: Maximum admissible SCOD risk :math:`\tau`.
        ood_cost: Relative cost :math:`c_{\mathrm{OOD}}` of accepting an OOD
            sample. The cost of accepting a misclassified ID sample is
            ``1 - ood_cost``. Defaults to ``0.5``.
        kwargs: Additional keyword arguments passed to
            :class:`torchmetrics.Metric`.

    References:
        [1] `Xia & Bouganis. Augmenting Softmax Information for Selective
        Classification with Out-of-Distribution Data. ACCV, 2022.
        <https://openaccess.thecvf.com/content/ACCV2022/papers/Xia_Augmenting_Softmax_Information_for_Selective_Classification_with_Out-of-Distribution_Data_ACCV_2022_paper.pdf>`_.

        [2] `Narasimhan et al. Plugin Estimators for Selective Classification
        with Out-of-Distribution Detection.
        <https://arxiv.org/abs/2301.12386>`_.

    See Also:
        :class:`~torch_uncertainty.metrics.classification.SCODAURC`:
            Definition of the SCOD loss and selective risk.
        :class:`~torch_uncertainty.metrics.classification.CovAtxRisk`:
            Corresponding selective-classification metric.
    """

    def __init__(
        self,
        risk_threshold: float,
        ood_cost: float = 0.5,
        **kwargs,
    ) -> None:
        super().__init__(risk_threshold=risk_threshold, **kwargs)
        self._set_ood_cost(ood_cost)


class SCODCovAt5Risk(SCODCovAtxRisk):
    r"""Calculate the maximum coverage at 5% SCOD risk.

    This is the fixed-threshold variant of
    :class:`~torch_uncertainty.metrics.classification.SCODCovAtxRisk` with
    :math:`\tau=0.05`:

    .. math::

        \operatorname{SCOD\text{-}Cov@5Risk}
        =
        \max\left\{
            \kappa_k :
            r(\kappa_k)\leq 0.05,\quad k\in\{1,\ldots,N\}
        \right\}.

    If no positive coverage has a SCOD risk at most 5%, the metric returns
    ``nan``.

    Args:
        ood_cost: Relative cost :math:`c_{\mathrm{OOD}}` of accepting an OOD
            sample. The cost of accepting a misclassified ID sample is
            ``1 - ood_cost``. Defaults to ``0.5``.
        kwargs: Additional keyword arguments passed to
            :class:`torchmetrics.Metric`.

    References:
        [1] `Xia & Bouganis. Augmenting Softmax Information for Selective
        Classification with Out-of-Distribution Data. ACCV, 2022.
        <https://openaccess.thecvf.com/content/ACCV2022/papers/Xia_Augmenting_Softmax_Information_for_Selective_Classification_with_Out-of-Distribution_Data_ACCV_2022_paper.pdf>`_.

        [2] `Narasimhan et al. Plugin Estimators for Selective Classification
        with Out-of-Distribution Detection.
        <https://arxiv.org/abs/2301.12386>`_.

    See Also:
        :class:`~torch_uncertainty.metrics.classification.SCODCovAtxRisk`:
            General metric with a configurable risk threshold.
    """

    def __init__(self, ood_cost: float = 0.5, **kwargs) -> None:
        super().__init__(
            risk_threshold=0.05,
            ood_cost=ood_cost,
            **kwargs,
        )


class SCODRiskAtxCov(_SCODRiskCoverageMixin, RiskAtxCov):
    r"""Calculate the SCOD risk at a specified coverage.

    This metric applies
    :class:`~torch_uncertainty.metrics.classification.RiskAtxCov` to the joint
    SCOD loss defined by
    :class:`~torch_uncertainty.metrics.classification.SCODAURC`.

    For a target coverage :math:`\gamma\in[0,1]`, let

    .. math::

        k_\gamma = \left\lceil \gamma N \right\rceil.

    Samples are ordered from lowest to highest OOD score, and the reported
    SCOD risk is

    .. math::

        \operatorname{SCOD\text{-}Risk@Cov}(\gamma)
        =
        r\left(\frac{k_\gamma}{N}\right)
        =
        \frac{1}{k_\gamma}
        \sum_{j=1}^{k_\gamma}\ell_{\sigma(j)}.

    As input to ``forward`` and ``update``, the metric accepts:

    - **ood_scores** (:class:`~torch.Tensor`): Float tensor containing OOD
      scores, where larger values indicate more OOD-like samples.
    - **classification_errors** (:class:`~torch.Tensor`): Boolean or binary
      tensor indicating misclassified ID samples.
    - **is_ood** (:class:`~torch.Tensor`): Boolean or binary tensor indicating
      OOD samples.

    As output to ``forward`` and ``compute``, the metric returns:

    - **scod_risk** (:class:`~torch.Tensor`): Scalar tensor containing the SCOD
      risk at the requested coverage. Lower values are better.

    Args:
        cov_threshold: Target coverage :math:`\gamma`.
        ood_cost: Relative cost :math:`c_{\mathrm{OOD}}` of accepting an OOD
            sample. The cost of accepting a misclassified ID sample is
            ``1 - ood_cost``. Defaults to ``0.5``.
        kwargs: Additional keyword arguments passed to
            :class:`torchmetrics.Metric`.

    References:
        [1] `Xia & Bouganis. Augmenting Softmax Information for Selective
        Classification with Out-of-Distribution Data. ACCV, 2022.
        <https://openaccess.thecvf.com/content/ACCV2022/papers/Xia_Augmenting_Softmax_Information_for_Selective_Classification_with_Out-of-Distribution_Data_ACCV_2022_paper.pdf>`_.

        [2] `Narasimhan et al. Plugin Estimators for Selective Classification
        with Out-of-Distribution Detection.
        <https://arxiv.org/abs/2301.12386>`_.

    See Also:
        :class:`~torch_uncertainty.metrics.classification.SCODAURC`:
            Definition of the SCOD loss and selective risk.
        :class:`~torch_uncertainty.metrics.classification.RiskAtxCov`:
            Corresponding selective-classification metric.
    """

    def __init__(
        self,
        cov_threshold: float,
        ood_cost: float = 0.5,
        **kwargs,
    ) -> None:
        super().__init__(cov_threshold=cov_threshold, **kwargs)
        self._set_ood_cost(ood_cost)


class SCODRiskAt80Cov(SCODRiskAtxCov):
    r"""Calculate the SCOD risk at 80% coverage.

    This is the fixed-coverage variant of
    :class:`~torch_uncertainty.metrics.classification.SCODRiskAtxCov` with
    :math:`\gamma=0.8`. For :math:`N` samples, it reports

    .. math::

        \operatorname{SCOD\text{-}Risk@80Cov}
        =
        r\left(
            \frac{\lceil 0.8N\rceil}{N}
        \right).

    Samples are ordered from lowest to highest OOD score before the risk is
    evaluated.

    Args:
        ood_cost: Relative cost :math:`c_{\mathrm{OOD}}` of accepting an OOD
            sample. The cost of accepting a misclassified ID sample is
            ``1 - ood_cost``. Defaults to ``0.5``.
        kwargs: Additional keyword arguments passed to
            :class:`torchmetrics.Metric`.

    References:
        [1] `Xia & Bouganis. Augmenting Softmax Information for Selective
        Classification with Out-of-Distribution Data. ACCV, 2022.
        <https://openaccess.thecvf.com/content/ACCV2022/papers/Xia_Augmenting_Softmax_Information_for_Selective_Classification_with_Out-of-Distribution_Data_ACCV_2022_paper.pdf>`_.

        [2] `Narasimhan et al. Plugin Estimators for Selective Classification
        with Out-of-Distribution Detection.
        <https://arxiv.org/abs/2301.12386>`_.

    See Also:
        :class:`~torch_uncertainty.metrics.classification.SCODRiskAtxCov`:
            General metric with configurable coverage.
    """

    def __init__(self, ood_cost: float = 0.5, **kwargs) -> None:
        super().__init__(
            cov_threshold=0.8,
            ood_cost=ood_cost,
            **kwargs,
        )
