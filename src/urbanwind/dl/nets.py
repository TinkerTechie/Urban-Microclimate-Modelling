"""
nets.py: the network architectures, and the registry every stage builds its models through.

Every layer that looks at neighbouring cells wraps around the domain edges (circular padding,
periodic upsampling), because the simulated city repeats like a tiled floor. A 4-level network
with this padding is exactly shift-equivariant for shifts that are multiples of 16 cells.

Public API
----------
UNet(in_ch, ...)                  : the S11 baseline
register(name)                    : decorator that adds an architecture to the registry
build_model(name, in_ch, **kw)    : build any registered architecture
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn

MODELS: dict[str, Callable[..., nn.Module]] = {}


def register(name: str) -> Callable:
    """Add an architecture to the registry so configs can name it (S12a adds its own this way)."""
    def add(cls):
        if name in MODELS:
            raise ValueError(f"model {name!r} is already registered")
        MODELS[name] = cls
        return cls
    return add


def build_model(name: str, in_ch: int, **kwargs) -> nn.Module:
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}; registered: {sorted(MODELS)}")
    return MODELS[name](in_ch=in_ch, **kwargs)


def conv_block(cin: int, cout: int, groups: int = 8) -> nn.Sequential:
    """Two 3x3 convolutions, each followed by GroupNorm and ReLU, with circular padding."""
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, padding_mode="circular"), nn.GroupNorm(groups, cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, padding_mode="circular"), nn.GroupNorm(groups, cout), nn.ReLU(inplace=True),
    )


def upsample_periodic(x: torch.Tensor) -> torch.Tensor:
    """Bilinear 2x upsampling that wraps around the edges.

    Plain bilinear upsampling clamps at the border; padding one cell circularly first and cropping
    afterwards makes the edge cells blend with their periodic neighbours instead.
    """
    up = F.interpolate(F.pad(x, (1, 1, 1, 1), mode="circular"), scale_factor=2, mode="bilinear", align_corners=False)
    return up[..., 2:-2, 2:-2]


@register("unet")
class UNet(nn.Module):
    """Periodic U-Net: 4 levels, GroupNorm, spatial dropout, and a softplus output so speeds stay positive."""

    def __init__(self, in_ch: int = 5, widths: tuple[int, ...] = (32, 64, 128, 256), bottleneck: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        self.down = nn.ModuleList()
        c = in_ch
        for w in widths:
            self.down.append(conv_block(c, w))
            c = w
        self.mid = conv_block(c, bottleneck)
        self.up_conv, self.up_block = nn.ModuleList(), nn.ModuleList()
        c = bottleneck
        for w in reversed(widths):
            self.up_conv.append(nn.Conv2d(c, w, 3, padding=1, padding_mode="circular"))
            self.up_block.append(conv_block(2 * w, w))
            c = w
        # spatial dropout drops whole feature maps; S14 keeps it on at prediction time for MC dropout
        self.drop = nn.Dropout2d(dropout)
        self.head = nn.Conv2d(c, 1, 1)
        self.multiple = 2 ** len(widths)
        for m in self.modules():  # He initialisation, suited to ReLU layers
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2] % self.multiple or x.shape[-1] % self.multiple:
            raise ValueError(f"input sides {tuple(x.shape[-2:])} must be multiples of {self.multiple}")
        skips = []
        for block in self.down:
            x = block(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.drop(self.mid(x))
        last = len(self.up_block) - 1
        for i, (conv, block, skip) in enumerate(zip(self.up_conv, self.up_block, reversed(skips))):
            x = block(torch.cat([conv(upsample_periodic(x)), skip], dim=1))
            if i < last:  # no dropout right before the output layer, which only maps features to speed
                x = self.drop(x)
        return F.softplus(self.head(x))
