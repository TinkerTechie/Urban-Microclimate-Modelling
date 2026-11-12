"""
nets.py: the network architectures, and the registry every stage builds its models through.

Every layer that looks at neighbouring cells wraps around the domain edges (circular padding,
periodic upsampling), because the simulated city repeats like a tiled floor. A 4-level network
with this padding is exactly shift-equivariant for shifts that are multiples of 16 cells.

Public API
----------
UNet(in_ch, ...)                  : the S11 baseline
ConvNeXtUNet(in_ch, ...)          : S13: the same U-Net layout built from ConvNeXt blocks
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


class LayerNorm2d(nn.Module):
    """LayerNorm over the channels of each cell separately, as in ConvNeXt.

    Unlike GroupNorm it never averages over space, so a cell's output does not depend on how large
    the patch or the domain is.
    """

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # statistics in float32 even under mixed precision; written out rather than permuting to
        # channels-last for F.layer_norm, which copies the whole tensor twice
        x32 = x.float()
        mean = x32.mean(dim=1, keepdim=True)
        var = (x32 - mean).pow(2).mean(dim=1, keepdim=True)
        out = (x32 - mean) * torch.rsqrt(var + self.eps)
        return (out * self.weight[:, None, None] + self.bias[:, None, None]).to(x.dtype)


class ConvNeXtBlock(nn.Module):
    """ConvNeXt block (Liu et al. 2022): a 7x7 depthwise convolution with circular padding, LayerNorm,
    a 1x1 expansion to 4x the channels with GELU, a 1x1 projection back, and a residual connection.

    The 7x7 filter sees a wider neighbourhood than the U-Net's 3x3 ones at every level. The layer
    scale starts small, so each block begins close to the identity.
    """

    def __init__(self, channels: int, kernel: int = 7, expansion: int = 4, layer_scale: float = 1e-6):
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, kernel, padding=kernel // 2, groups=channels, padding_mode="circular")
        self.norm = LayerNorm2d(channels)
        self.pw1 = nn.Conv2d(channels, expansion * channels, 1)
        self.pw2 = nn.Conv2d(expansion * channels, channels, 1)
        self.gamma = nn.Parameter(torch.full((channels, 1, 1), layer_scale))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.gamma * self.pw2(F.gelu(self.pw1(self.norm(self.dw(x)))))


def convnext_stage(channels: int, depth: int) -> nn.Sequential:
    return nn.Sequential(*(ConvNeXtBlock(channels) for _ in range(depth)))


@register("convnext_unet")
class ConvNeXtUNet(nn.Module):
    """Periodic U-Net built from ConvNeXt blocks, with the S11 U-Net's layout and output.

    - Encoder: a 3x3 circular stem, then at each level ConvNeXt blocks, a skip connection, and a
      LayerNorm + 2x2 stride-2 convolution that halves the grid (no padding needed).
    - Decoder: periodic bilinear upsampling, a 1x1 convolution to the level's width, concatenation
      with the skip, a 1x1 convolution that merges them, and ConvNeXt blocks.
    - Output: LayerNorm, 1x1 convolution and softplus, so speeds stay positive. Spatial dropout
      sits where the U-Net has it, so MC dropout works the same way.
    No positional or coordinate input, and every layer that looks at neighbours wraps around.
    """

    def __init__(self, in_ch: int = 5, widths: tuple[int, ...] = (32, 64, 128, 256), bottleneck: int = 384,
                 depths: tuple[int, ...] = (2, 2, 2, 2), mid_depth: int = 2, up_depth: int = 1,
                 dropout: float = 0.08):
        super().__init__()
        if len(depths) != len(widths):
            raise ValueError("depths needs one entry per level")
        self.stem = nn.Conv2d(in_ch, widths[0], 3, padding=1, padding_mode="circular")
        self.down_blocks, self.downsample = nn.ModuleList(), nn.ModuleList()
        nxt = [*widths[1:], bottleneck]
        for w, d, w_next in zip(widths, depths, nxt):
            self.down_blocks.append(convnext_stage(w, d))
            self.downsample.append(nn.Sequential(LayerNorm2d(w), nn.Conv2d(w, w_next, 2, stride=2)))
        self.mid = convnext_stage(bottleneck, mid_depth)
        self.up_proj, self.merge, self.up_blocks = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        c = bottleneck
        for w in reversed(widths):
            self.up_proj.append(nn.Conv2d(c, w, 1))
            self.merge.append(nn.Conv2d(2 * w, w, 1))
            self.up_blocks.append(convnext_stage(w, up_depth))
            c = w
        self.drop = nn.Dropout2d(dropout)
        self.head = nn.Sequential(LayerNorm2d(c), nn.Conv2d(c, 1, 1))
        self.multiple = 2 ** len(widths)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2] % self.multiple or x.shape[-1] % self.multiple:
            raise ValueError(f"input sides {tuple(x.shape[-2:])} must be multiples of {self.multiple}")
        x, skips = self.stem(x), []
        for blocks, down in zip(self.down_blocks, self.downsample):
            x = blocks(x)
            skips.append(x)
            x = down(x)
        x = self.drop(self.mid(x))
        last = len(self.up_blocks) - 1
        for i, (proj, merge, blocks, skip) in enumerate(zip(self.up_proj, self.merge, self.up_blocks, reversed(skips))):
            x = blocks(merge(torch.cat([proj(upsample_periodic(x)), skip], dim=1)))
            if i < last:
                x = self.drop(x)
        return F.softplus(self.head(x))
