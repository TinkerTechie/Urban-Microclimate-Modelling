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


class ResBlock(nn.Module):
    """Residual block: two circular 3x3 convolutions with GroupNorm and ReLU, plus 1x1 projection."""

    def __init__(self, cin: int, cout: int, groups: int = 8):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1, padding_mode="circular")
        self.gn1 = nn.GroupNorm(groups, cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, padding_mode="circular")
        self.gn2 = nn.GroupNorm(groups, cout)
        self.proj = nn.Conv2d(cin, cout, 1) if cin != cout else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.proj(x) if self.proj is not None else x
        h = F.relu(self.gn1(self.conv1(x)), inplace=True)
        h = self.gn2(self.conv2(h))
        return F.relu(h + res, inplace=True)


@register("res_unet")
class ResUNet(nn.Module):
    """Periodic Res-U-Net: S11 layout with residual blocks and softplus output."""

    def __init__(self, in_ch: int = 5, widths: tuple[int, ...] = (24, 48, 96, 192), bottleneck: int = 384,
                 dropout: float = 0.08):
        super().__init__()
        self.down = nn.ModuleList()
        c = in_ch
        for w in widths:
            self.down.append(ResBlock(c, w))
            c = w
        self.mid = ResBlock(c, bottleneck)
        self.up_conv, self.up_block = nn.ModuleList(), nn.ModuleList()
        c = bottleneck
        for w in reversed(widths):
            self.up_conv.append(nn.Conv2d(c, w, 3, padding=1, padding_mode="circular"))
            self.up_block.append(ResBlock(2 * w, w))
            c = w
        self.drop = nn.Dropout2d(dropout)
        self.head = nn.Conv2d(c, 1, 1)
        self.multiple = 2 ** len(widths)
        for m in self.modules():
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
            if i < last:
                x = self.drop(x)
        return F.softplus(self.head(x))


class AttentionGate(nn.Module):
    """Attention gate on skip connections (Oktay et al. 2018).

    Gating signal from the coarser level is brought to the skip's size with periodic upsampling.
    1x1 convolutions compute linear embeddings, followed by GroupNorm, ReLU, and sigmoid coefficient map.
    """

    def __init__(self, in_skip: int, in_g: int, int_ch: int, groups: int = 8):
        super().__init__()
        self.w_x = nn.Sequential(nn.Conv2d(in_skip, int_ch, 1), nn.GroupNorm(groups, int_ch))
        self.w_g = nn.Sequential(nn.Conv2d(in_g, int_ch, 1), nn.GroupNorm(groups, int_ch))
        self.psi = nn.Conv2d(int_ch, 1, 1)

    def forward(self, skip: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        g_up = upsample_periodic(g)
        act = F.relu(self.w_x(skip) + self.w_g(g_up), inplace=True)
        coeff = torch.sigmoid(self.psi(act))
        return skip * coeff


@register("attn_unet")
class AttnUNet(nn.Module):
    """Periodic Attention U-Net: S11 layout with attention gates on skip connections."""

    def __init__(self, in_ch: int = 5, widths: tuple[int, ...] = (24, 48, 96, 192), bottleneck: int = 384,
                 dropout: float = 0.08):
        super().__init__()
        self.down = nn.ModuleList()
        c = in_ch
        for w in widths:
            self.down.append(conv_block(c, w))
            c = w
        self.mid = conv_block(c, bottleneck)
        self.up_conv, self.up_block, self.gates = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        c = bottleneck
        for w in reversed(widths):
            self.up_conv.append(nn.Conv2d(c, w, 3, padding=1, padding_mode="circular"))
            self.gates.append(AttentionGate(in_skip=w, in_g=c, int_ch=w))
            self.up_block.append(conv_block(2 * w, w))
            c = w
        self.drop = nn.Dropout2d(dropout)
        self.head = nn.Conv2d(c, 1, 1)
        self.multiple = 2 ** len(widths)
        for m in self.modules():
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
        for i, (conv, block, gate, skip) in enumerate(zip(self.up_conv, self.up_block, self.gates, reversed(skips))):
            gated_skip = gate(skip, x)
            x = block(torch.cat([conv(upsample_periodic(x)), gated_skip], dim=1))
            if i < last:
                x = self.drop(x)
        return F.softplus(self.head(x))


class SpectralBlock(nn.Module):
    """2D Spectral convolution block for Fourier Neural Operator (FNO).

    Runs 2D real FFT, retains low frequency modes, applies learned complex weights, and maps back
    with 2D inverse real FFT. The FFT and spectral multiplication execute in float32 with autocast
    disabled to ensure compatibility with mixed precision training and arbitrary spatial tile sizes.
    Adds a 1x1 convolution bypass followed by GELU activation.
    """

    def __init__(self, width: int, modes1: int = 16, modes2: int = 16):
        super().__init__()
        self.width = width
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1.0 / (width * width)
        self.weights1 = nn.Parameter(scale * torch.randn(width, width, modes1, modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(scale * torch.randn(width, width, modes1, modes2, dtype=torch.cfloat))
        self.bypass = nn.Conv2d(width, width, 1)

    def _apply(self, fn):
        super()._apply(fn)
        if self.bypass.weight.dtype == torch.float64:
            self.weights1.data = self.weights1.data.to(torch.cdouble)
            self.weights2.data = self.weights2.data.to(torch.cdouble)
        elif self.bypass.weight.dtype == torch.float32:
            self.weights1.data = self.weights1.data.to(torch.cfloat)
            self.weights2.data = self.weights2.data.to(torch.cfloat)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        bypass = self.bypass(x)
        with torch.autocast(x.device.type, enabled=False):
            c_dtype = self.weights1.dtype
            r_dtype = torch.float64 if c_dtype == torch.cdouble else torch.float32
            x_in = x.to(dtype=r_dtype)
            x_ft = torch.fft.rfft2(x_in)
            out_ft = torch.zeros(b, self.width, h, w // 2 + 1, device=x.device, dtype=c_dtype)
            m1 = min(self.modes1, h // 2)
            m2 = min(self.modes2, w // 2)
            out_ft[:, :, :m1, :m2] = torch.einsum(
                "bixy,ioxy->boxy", x_ft[:, :, :m1, :m2], self.weights1[:, :, :m1, :m2]
            )
            out_ft[:, :, -m1:, :m2] = torch.einsum(
                "bixy,ioxy->boxy", x_ft[:, :, -m1:, :m2], self.weights2[:, :, :m1, :m2]
            )
            spec = torch.fft.irfft2(out_ft, s=(h, w)).to(dtype=bypass.dtype)
        return F.gelu(spec + bypass)


@register("fno")
class FNO(nn.Module):
    """Fourier Neural Operator (FNO): 1x1 lifting, 4 spectral layers with bypass, 1x1 projection.

    Modes are counted in grid cells (for example, mode 16 corresponds to a 64 m wave on a 1,024-cell
    domain but a 16 m wave on a 256-cell patch). To preserve physical wavelength correspondence,
    inference should use 256x256 wrap-around tiles matching training (infer_window: 256, infer_border: 16).
    Contains no positional or coordinate channels and no domain padding. Multiple is 1.
    """

    def __init__(self, in_ch: int = 5, width: int = 48, modes1: int = 16, modes2: int = 16, n_layers: int = 4,
                 dropout: float = 0.08):
        super().__init__()
        self.multiple = 1
        self.lift = nn.Conv2d(in_ch, width, 1)
        self.spectral_layers = nn.ModuleList([
            SpectralBlock(width, modes1, modes2) for _ in range(n_layers)
        ])
        self.drop = nn.Dropout2d(dropout)
        self.proj = nn.Conv2d(width, 1, 1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lift(x)
        last = len(self.spectral_layers) - 1
        for i, layer in enumerate(self.spectral_layers):
            x = layer(x)
            if i < last:
                x = self.drop(x)
        return F.softplus(self.proj(x))

