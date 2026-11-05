"""
losses.py: training losses. Every loss has the signature ``loss(pred, y, mask) -> scalar tensor``.

The mean is taken over the valid cells of the whole batch, so a patch full of buildings does not
count less per cell than an open one, and a batch without any valid cell gives 0 instead of NaN.
Losses are computed in float32 even when the network runs in half precision.

Public API
----------
masked_l1(pred, y, mask)  : mean absolute error over valid cells (learns the median of each cell's speed)
masked_mse(pred, y, mask) : mean squared error over valid cells (learns the mean; matches R² and RMSE)
LOSSES                    : name -> loss, for configs (S12b adds its losses here)
"""

from __future__ import annotations

import torch


def masked_l1(pred: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.float()
    error = (pred.float() - y.float()).abs() * mask
    return error.sum() / mask.sum().clamp(min=1.0)


def masked_mse(pred: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.float()
    error = (pred.float() - y.float()) ** 2 * mask
    return error.sum() / mask.sum().clamp(min=1.0)


LOSSES = {"l1": masked_l1, "mse": masked_mse}
