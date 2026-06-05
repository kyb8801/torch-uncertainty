# ruff: noqa: D212, D415, T201
"""
DEUP (Direct Epistemic Uncertainty Prediction) with TorchUncertainty
====================================================================

Minimal example: train an error predictor on out-of-fold calibration errors and
use epistemic scores for OOD detection.

For sklearn / tabular / time-series DEUP, use the standalone package:
https://github.com/ursinasanderink/deup
"""

# %%
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from torch_uncertainty.post_processing import DEUP

# %%
# 1. Synthetic classification task
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

torch.manual_seed(0)
n_train, n_cal, in_dim, n_classes = 200, 100, 8, 5
x_train = torch.randn(n_train, in_dim)
y_train = torch.randint(0, n_classes, (n_train,))
x_cal = torch.randn(n_cal, in_dim)
y_cal = torch.randint(0, n_classes, (n_cal,))

model = nn.Sequential(nn.Linear(in_dim, 32), nn.ReLU(), nn.Linear(32, n_classes))
# pre-trained weights would come from ClassificationRoutine.fit in practice

cal_loader = DataLoader(TensorDataset(x_cal, y_cal), batch_size=32)

# %%
# 2. Fit DEUP on the calibration split (OOF errors, Algorithm 2)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

deup = DEUP(
    task="classification",
    model=model,
    n_folds=5,
    hidden_dim=32,
    max_epochs=30,
    device="cpu",
)
deup.fit(cal_loader)

# %%
# 3. Epistemic uncertainty at inference
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

x_test = torch.randn(10, in_dim)
uncertainty = deup(x_test)
print("Epistemic g(x):", uncertainty)

probs = deup.predict_proba(x_test)
print("Base-model probs shape:", probs.shape)

# %%
# 4. Use with ClassificationRoutine + OOD evaluation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# .. code-block:: python
#
#     from torch_uncertainty.routines import ClassificationRoutine
#
#     baseline = ClassificationRoutine(
#         num_classes=10,
#         model=model,
#         post_processing=deup,
#         ood_criterion="deup",
#         eval_ood=True,
#     )
#     trainer.test(baseline, datamodule=datamodule)
