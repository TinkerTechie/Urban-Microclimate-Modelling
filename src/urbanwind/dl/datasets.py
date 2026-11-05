"""
datasets.py: training patches and full-domain prediction on top of the S10 cache.

Public API
----------
wrap_window(a, r0, c0, h, w)  : an h×w window of a periodic field, wrapping across its edges
mirror_patch(x, y, mask)      : mirror a patch across the wind axis
PatchDataset(cases, ...)      : random 256×256 training patches as (x, y, mask) tensors
predict_arrays(model, arrays) : U/u_tau on the native grid for one case held in memory
predict_native(model, case)   : the same for a cached development case
"""

from __future__ import annotations

import math
import pathlib
from typing import Any

import numpy as np
import torch
from scipy import ndimage
from torch.utils.data import Dataset

from urbanwind.dl.cache import (BUILDING_CHANNEL, CACHE_DIR, FINE_DX, SIN_CHANNEL, from_canonical, load_cached,
                                make_inputs, read_cached)

PATCH = 256
BORDER = 16
MARGIN = 64  # periodic context added around a whole domain before prediction
MULTIPLE = 16  # four 2x poolings in the U-Net need sides divisible by 16


def wrap_window(a: np.ndarray, r0: int, c0: int, h: int, w: int) -> np.ndarray:
    """The h×w window of a periodic field whose top-left cell is (r0, c0), over the last two axes.

    The indices wrap around the domain, so a window that runs off one edge continues from the
    opposite edge, and a window larger than the domain simply tiles it.
    """
    ny, nx = a.shape[-2:]
    rows = (r0 + np.arange(h)) % ny
    cols = (c0 + np.arange(w)) % nx
    return a[..., rows[:, None], cols[None, :]]


def mirror_patch(x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mirror a patch across the wind axis: flip the rows and flip the sign of the wind angle.

    Flipping the rows sends a wind angle a to -a, so only the sine channel changes sign. Without
    Coriolis forcing the mirrored case is as valid a flow as the original. This is exact for a
    scalar target such as U; a target with a v component would also need its sign flipped.
    """
    x = x[:, ::-1].copy()
    x[SIN_CHANNEL] *= -1.0
    return x, y[::-1].copy(), mask[::-1].copy()


class PatchDataset(Dataset):
    """Random square training patches cut from whole cached cases.

    Each item draws a case with probability proportional to the square root of its open area
    (so large cities do not drown out the small idealized layouts), a random top-left cell, and
    with ``mirror`` a coin flip for the mirror. Items depend only on ``(seed, epoch, index)``, so
    a run is reproducible and stays correct with several DataLoader workers. Set ``epoch`` before
    each epoch to get fresh patches; with ``persistent_workers=True`` the workers keep their own
    copy and never see the new epoch, so leave that option off.

    The network pads every patch circularly, which glues its top edge to its bottom edge even
    though they are not neighbours in the city. ``border`` cells along each side are therefore
    left out of the loss mask.

    Only the stored float16 height, wall distance and target are kept in memory (about 0.7 GB for
    all 482 cases); the five input channels are built for each patch.
    """

    def __init__(self, cases: list[str], patch: int = PATCH, mirror: bool = True, seed: int = 0,
                 border: int = BORDER, cache_dir: pathlib.Path = CACHE_DIR):
        if not cases:
            raise ValueError("PatchDataset needs at least one case")
        if not 0 <= 2 * border < patch:
            raise ValueError(f"border {border} leaves nothing of a {patch} patch")
        self.patch, self.mirror, self.seed, self.border, self.epoch = patch, mirror, seed, border, 0
        self.cases = sorted(cases)  # the patches then do not depend on the order the cases came in
        self.items = [{"H": c["H"], "dist": c["dist"], "y": np.nan_to_num(c["U"], nan=0.0), "valid": c["valid"],
                       "angle_deg": c["angle_deg"]} for c in (read_cached(case, cache_dir) for case in self.cases)]
        open_area = np.array([item["valid"].sum() for item in self.items], dtype=float)
        self.weights = np.sqrt(open_area) / np.sqrt(open_area).sum()
        # one epoch holds as many patches as it takes to cover the cases' total area once
        self.length = math.ceil(sum(item["valid"].size for item in self.items) / patch**2)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rng = np.random.default_rng((self.seed, self.epoch, index))
        item = self.items[rng.choice(len(self.items), p=self.weights)]
        ny, nx = item["valid"].shape
        r0, c0 = int(rng.integers(ny)), int(rng.integers(nx))
        H, dist, y, mask = (wrap_window(item[name], r0, c0, self.patch, self.patch) for name in ("H", "dist", "y", "valid"))
        x, y = make_inputs(H, dist, item["angle_deg"]), y.astype(np.float32)
        if self.border:
            b = self.border
            mask[:b], mask[-b:], mask[:, :b], mask[:, -b:] = False, False, False, False
        if self.mirror and rng.random() < 0.5:
            x, y, mask = mirror_patch(x, y, mask)
        return torch.from_numpy(x), torch.from_numpy(y[None]), torch.from_numpy(mask[None])


def padded_size(n: int) -> tuple[int, int]:
    """(cells added before, total length) for one side: at least MARGIN on each side, rounded up to MULTIPLE."""
    total = math.ceil((n + 2 * MARGIN) / MULTIPLE) * MULTIPLE
    return (total - n) // 2, total


def predict_arrays(model: torch.nn.Module, c: dict[str, Any], device: torch.device | str | None = None,
                   fill: bool = True) -> np.ndarray:
    """U/u_tau for one whole case, on the grid and in the orientation of the raw files.

    ``c`` holds the arrays of one case with its inputs: ``load_cached(case)`` for a development
    case, or ``with_inputs(build_arrays(load_case(case)))`` for a test case at final scoring, so
    test cases never need to enter the cache. Only ``x`` and the transform are read.

    The domain is wrapped periodically by at least MARGIN cells on every side, so the seam that
    circular padding makes at the edge of the input lies outside the city. The prediction is
    cropped back, turned back out of the canonical frame, and for 0.5 m cases every 1 m value is
    copied to its four sub-cells. The result lines up cell by cell with ``load_case(case)["U"]``,
    which is what ``urbanwind.evaluate`` compares against.

    With ``fill`` every cell that the input marks as a building takes the prediction of the
    nearest open cell. The loss never sees building cells, so the raw output there is untrained,
    yet in a 0.5 m case a 1 m block that straddles a wall still holds open sub-cells that are
    scored (the S10 README gives how many). The fill uses the input only, never the simulation.

    The caller chooses ``model.eval()`` or ``model.train()`` (S14 keeps dropout on for MC dropout).
    """
    _, ny, nx = c["x"].shape
    top, h = padded_size(ny)
    left, w = padded_size(nx)
    x = wrap_window(c["x"], -top, -left, h, w)
    if device is None:
        device = next(model.parameters()).device
    with torch.no_grad():
        out = model(torch.from_numpy(x)[None].to(device))[0, 0].float().cpu().numpy()
    if fill:
        # done on the padded window, so the nearest open cell may lie across the periodic edge
        buildings = x[BUILDING_CHANNEL] > 0
        nearest = ndimage.distance_transform_edt(buildings, return_distances=False, return_indices=True)
        out = out[nearest[0], nearest[1]]
    y = from_canonical(out[top:top + ny, left:left + nx], c["k"], c["transposed"])
    if np.isclose(c["dx_native"], FINE_DX):
        y = y.repeat(2, axis=0).repeat(2, axis=1)
    if y.shape != tuple(c["native_shape"]):
        raise ValueError(f"prediction shape {y.shape} does not match the native grid {tuple(c['native_shape'])}")
    return np.ascontiguousarray(y)


def predict_native(model: torch.nn.Module, case: str, cache_dir: pathlib.Path = CACHE_DIR,
                   device: torch.device | str | None = None, fill: bool = True) -> np.ndarray:
    """``predict_arrays`` for a cached development case."""
    return predict_arrays(model, load_cached(case, cache_dir), device, fill)
