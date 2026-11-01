"""
DIPCYT-X40s — Model
====================
Architecture extracted verbatim from coco-dipcyt-x40s.ipynb cells 23, 27, 29, 32, 36.
Logic is unchanged; notebook diagnostics/prints removed.

Components
----------
- TPHYOLOv5PlusPlus  (backbone: Focus → PartialConv → CSPDarknet53 + C3TR)
- FPCBAM             (Fourier Positional Convolutional Block Attention Module)
- PANet              (Path Aggregation Network with Swin Transformer blocks)
- TransformerPolygonHead  (encoder + decoder + anchor-relative polygon regression)
- FullModel          (backbone → PANet → 4 heads on D1-D4)
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  BACKBONE  — TPHYOLOv5PlusPlus
# ═══════════════════════════════════════════════════════════════════════════════

class StdConvBlock(nn.Module):
    """Conv2d + GroupNorm + SiLU."""
    def __init__(self, in_c: int, out_c: int, k: int = 3, s: int = 1, p: int = 1):
        super().__init__()
        num_groups = min(32, out_c) if out_c % min(32, out_c) == 0 else 1
        self.conv = nn.Conv2d(in_c, out_c, k, s, p, bias=False)
        self.norm = nn.GroupNorm(num_groups, out_c)
        self.act  = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class PartialConv2d(nn.Module):
    """Partial convolution for handling occluded/missing regions.
    Mask defaults to all-ones (no occlusion) when not provided."""
    def __init__(self, in_c: int, out_c: int, k: int = 3, s: int = 1, p: int = 1):
        super().__init__()
        num_groups = min(32, out_c) if out_c % min(32, out_c) == 0 else 1
        self.conv      = nn.Conv2d(in_c, out_c, k, s, p, bias=False)
        self.mask_conv  = nn.Conv2d(in_c, out_c, k, s, p, bias=False)
        nn.init.constant_(self.mask_conv.weight, 1.0)
        for p_ in self.mask_conv.parameters():
            p_.requires_grad = False
        self.norm = nn.GroupNorm(num_groups, out_c)
        self.act  = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is None:
            mask = torch.ones_like(x)
        with torch.no_grad():
            mask_out = self.mask_conv(mask)
        out = self.conv(x * mask) / mask_out.clamp(min=1e-4)
        return self.act(self.norm(out))


class PConvBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, k: int = 3, s: int = 1, p: int = 1):
        super().__init__()
        self.pconv = PartialConv2d(in_c, out_c, k, s, p)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.pconv(x, mask)


class Focus(nn.Module):
    """Space-to-depth (YOLOv5 Focus): 4× sub-sampling via pixel shuffle → 1×1 conv."""
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.conv = StdConvBlock(in_c * 4, out_c, k=1, s=1, p=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat([
            x[..., 0::2, 0::2],
            x[..., 1::2, 0::2],
            x[..., 0::2, 1::2],
            x[..., 1::2, 1::2],
        ], dim=1)
        return self.conv(x)


class Bottleneck(nn.Module):
    """Bottleneck with residual skip connection (matches YOLOv5)."""
    def __init__(self, channels: int):
        super().__init__()
        self.cv1 = StdConvBlock(channels, channels, k=1, s=1, p=0)
        self.cv2 = StdConvBlock(channels, channels, k=3, s=1, p=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.cv2(self.cv1(x))


class C3Block(nn.Module):
    """CSP Bottleneck with 3 convolutions."""
    def __init__(self, in_c: int, out_c: int, num_blocks: int = 1):
        super().__init__()
        mid         = out_c // 2
        self.cv1    = StdConvBlock(in_c,    mid,   k=1, s=1, p=0)
        self.cv2    = StdConvBlock(in_c,    mid,   k=1, s=1, p=0)
        self.cv3    = StdConvBlock(mid * 2, out_c, k=1, s=1, p=0)
        self.blocks = nn.Sequential(*[Bottleneck(mid) for _ in range(num_blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y1 = self.blocks(self.cv1(x))
        y2 = self.cv2(x)
        return self.cv3(torch.cat([y1, y2], dim=1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling — Fast."""
    def __init__(self, in_c: int, out_c: int, k: int = 5):
        super().__init__()
        mid       = in_c // 2
        self.cv1  = StdConvBlock(in_c,    mid,   k=1, s=1, p=0)
        self.cv2  = StdConvBlock(mid * 4, out_c, k=1, s=1, p=0)
        self.pool = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x  = self.cv1(x)
        p1 = self.pool(x)
        p2 = self.pool(p1)
        p3 = self.pool(p2)
        return self.cv2(torch.cat([x, p1, p2, p3], dim=1))


class TransformerEncoderBlock(nn.Module):
    """Pre-norm transformer encoder block used inside C3TR."""
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, num_heads, dropout=0.0, batch_first=True)
        self.ff    = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n           = self.norm1(x)
        attn_out, _ = self.attn(n, n, n)
        x           = x + attn_out
        x           = x + self.ff(self.norm2(x))
        return x


class C3TR(nn.Module):
    """CSP Bottleneck with Transformer encoder layers (replaces CNN bottleneck)."""
    def __init__(self, in_c: int, out_c: int, num_heads: int = 8, num_layers: int = 3):
        super().__init__()
        mid      = out_c // 2
        self.cv1 = StdConvBlock(in_c,    mid,   k=1, s=1, p=0)
        self.cv2 = StdConvBlock(in_c,    mid,   k=1, s=1, p=0)
        self.cv3 = StdConvBlock(mid * 2, out_c, k=1, s=1, p=0)
        self.trs = nn.Sequential(*[
            TransformerEncoderBlock(mid, num_heads) for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        y1     = self.cv1(x)
        tokens = y1.flatten(2).permute(0, 2, 1)
        tokens = self.trs(tokens)
        y1     = tokens.permute(0, 2, 1).reshape(B, -1, H, W)
        y2     = self.cv2(x)
        return self.cv3(torch.cat([y1, y2], dim=1))


class TPHYOLOv5PlusPlus(nn.Module):
    """TPH-YOLOv5++ backbone: Focus → 2×PartialConv → CSPDarknet53 → SPPF → C3TR.
    Outputs P3 (256, 40×40), P4 (512, 20×20), P5 (512, 20×20)."""
    def __init__(self):
        super().__init__()
        self.focus  = Focus(3, 32)
        self.pconv1 = PConvBlock(32,  64,  k=3, s=2, p=1)
        self.c3_1   = C3Block(64,  64,  num_blocks=3)
        self.pconv2 = PConvBlock(64,  128, k=3, s=2, p=1)
        self.c3_2   = C3Block(128, 128, num_blocks=9)
        self.conv1  = StdConvBlock(128, 256, k=3, s=2, p=1)
        self.c3_3   = C3Block(256, 256, num_blocks=9)
        self.conv2  = StdConvBlock(256, 512, k=3, s=2, p=1)
        self.sppf   = SPPF(512, 512, k=5)
        self.c3tr   = C3TR(512, 512, num_heads=8, num_layers=3)

    def forward(self, x: torch.Tensor):
        x  = self.focus(x)       # [B,  32, 320, 320]
        x  = self.pconv1(x)      # [B,  64, 160, 160]
        x  = self.c3_1(x)        # [B,  64, 160, 160]
        x  = self.pconv2(x)      # [B, 128,  80,  80]
        p2 = self.c3_2(x)        # [B, 128,  80,  80]
        x  = self.conv1(p2)      # [B, 256,  40,  40]
        p3 = self.c3_3(x)        # [B, 256,  40,  40]  → P3
        p4 = self.conv2(p3)      # [B, 512,  20,  20]  → P4
        x  = self.sppf(p4)       # [B, 512,  20,  20]
        p5 = self.c3tr(x)        # [B, 512,  20,  20]  → P5
        return p3, p4, p5


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  FPCBAM — Fourier Positional Convolutional Block Attention Module
# ═══════════════════════════════════════════════════════════════════════════════

class PAM(nn.Module):
    """Position Attention Module — pairwise spatial self-attention with learned α."""
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv_B  = nn.Conv2d(in_channels, in_channels // 8, 1, bias=False)
        self.conv_C  = nn.Conv2d(in_channels, in_channels // 8, 1, bias=False)
        self.conv_D  = nn.Conv2d(in_channels, in_channels,      1, bias=False)
        self.alpha   = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        N = H * W
        feat_B = self.conv_B(x).view(B, -1, N)
        feat_C = self.conv_C(x).view(B, -1, N)
        S      = self.softmax(torch.bmm(feat_B.permute(0, 2, 1), feat_C))
        feat_D = self.conv_D(x).view(B, C, N)
        out    = torch.bmm(feat_D, S.permute(0, 2, 1)).view(B, C, H, W)
        return self.alpha * out + x


class FPCBAM(nn.Module):
    """FFT-augmented Channel + Spatial + Position Attention.
    XLA-safe: no in-place ops on the FFT path.
    use_pam=False skips the N×N attention matrix (OOM guard at 40×40)."""

    def __init__(self, gate_channels: int, reduction_ratio: int = 16, use_pam: bool = True):
        super().__init__()
        reduced = max(gate_channels // reduction_ratio, 1)
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(gate_channels, reduced,       1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced,       gate_channels, 1, bias=False),
        )
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=12, dilation=4, bias=False),
            nn.GroupNorm(1, 1),
            nn.Sigmoid(),
        )
        self.pam     = PAM(gate_channels) if use_pam else None
        self.use_pam = use_pam
        self.gate    = nn.Parameter(torch.zeros(1))

    @staticmethod
    def _fft_feature(x: torch.Tensor) -> torch.Tensor:
        """High-pass filter via FFT: zero-out DC component, add residual."""
        x_f32 = x.float()
        freq   = torch.fft.rfft2(x_f32, norm='ortho')
        freq[:, :, 0, 0] = 0
        out    = torch.fft.irfft2(freq, s=x_f32.shape[-2:], norm='ortho')
        return (x_f32 + out).to(x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self._fft_feature(x)
        # Channel attention
        avg_ch = self.channel_mlp(f.mean(dim=(2, 3), keepdim=True))
        max_ch = self.channel_mlp(f.amax(dim=(2, 3), keepdim=True))
        f_cbam = f * torch.sigmoid(avg_ch + max_ch)
        # Spatial attention
        avg_sp = f_cbam.mean(dim=1, keepdim=True)
        max_sp = f_cbam.amax(dim=1, keepdim=True)
        f_cbam = f_cbam * self.spatial_att(torch.cat([avg_sp, max_sp], dim=1))
        # Position attention
        f_pam = self.pam(f) if self.use_pam else f
        # Gated residual (gate initialised to 0 → identity at init)
        return self.gate * (f_pam + f_cbam) + x


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  PANet + Swin Transformer
# ═══════════════════════════════════════════════════════════════════════════════

class BNSequential(nn.Module):
    """Conv + GroupNorm + ReLU (called BNSequential for historical reasons; uses GN)."""
    def __init__(self, in_c: int, out_c: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1, lateral: bool = False):
        super().__init__()
        k = 1 if lateral else kernel_size
        p = 0 if lateral else padding
        self.conv = nn.Conv2d(in_c, out_c, k, stride=stride, padding=p, bias=False)
        ng = 32 if out_c % 32 == 0 else (16 if out_c % 16 == 0 else 1)
        self.norm = nn.GroupNorm(ng, out_c)
        self.act  = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


def _window_partition(x: torch.Tensor, ws: int) -> torch.Tensor:
    B, H, W, C = x.shape
    x = x.view(B, H // ws, ws, W // ws, ws, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws, ws, C)


def _window_reverse(windows: torch.Tensor, ws: int, H: int, W: int) -> torch.Tensor:
    B = int(windows.shape[0] / (H * W / ws / ws))
    x = windows.view(B, H // ws, W // ws, ws, ws, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


class WindowAttention(nn.Module):
    """Multi-head self-attention within local windows (Swin Transformer)."""
    def __init__(self, dim: int, window_size: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, num_heads))
        ch = torch.arange(window_size)
        cw = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(ch, cw, indexing="ij"))
        cf = torch.flatten(coords, 1)
        rel = (cf[:, :, None] - cf[:, None, :]).permute(1, 2, 0).contiguous()
        rel[:, :, 0] += window_size - 1
        rel[:, :, 1] += window_size - 1
        rel[:, :, 0] *= 2 * window_size - 1
        self.register_buffer("rpi", rel.sum(-1))
        self.qkv  = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B_, N, C = x.shape
        qkv = (self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
               .permute(2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q * self.scale) @ k.transpose(-2, -1)
        bias = (self.relative_position_bias_table[self.rpi.view(-1)]
                .view(N, N, -1).permute(2, 0, 1))
        attn = attn + bias.unsqueeze(0)
        if mask is not None:
            nW   = mask.shape[0]
            attn = (attn.view(B_ // nW, nW, self.num_heads, N, N)
                    + mask.unsqueeze(1).unsqueeze(0))
            attn = attn.view(-1, self.num_heads, N, N)
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj(x)


class SwinTransformerBlock(nn.Module):
    """Single Swin Transformer block with optional shifted-window attention."""
    def __init__(self, dim: int, num_heads: int = 8, window_size: int = 5,
                 shift_size: int = 0, mlp_ratio: float = 4.0):
        super().__init__()
        self.window_size = window_size
        self.shift_size  = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        h          = int(dim * mlp_ratio)
        self.mlp   = nn.Sequential(nn.Linear(dim, h), nn.GELU(), nn.Linear(h, dim))

    def _mask(self, H: int, W: int, ws: int, ss: int, device: torch.device) -> torch.Tensor:
        img   = torch.zeros((1, H, W, 1), device=device)
        spans = (slice(0, -ws), slice(-ws, -ss), slice(-ss, None))
        cnt   = 0
        for h in spans:
            for w in spans:
                img[:, h, w, :] = cnt
                cnt += 1
        mw = _window_partition(img, ws).view(-1, ws * ws)
        am = mw.unsqueeze(1) - mw.unsqueeze(2)
        return am.masked_fill(am != 0, -100.0).masked_fill(am == 0, 0.0)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, L, C  = x.shape
        ws, ss   = self.window_size, self.shift_size
        if min(H, W) <= ws:
            ws, ss = min(H, W), 0
        shortcut = x
        x        = self.norm1(x).view(B, H, W, C)
        pad_r    = (ws - W % ws) % ws
        pad_b    = (ws - H % ws) % ws
        x        = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        Hp, Wp   = H + pad_b, W + pad_r
        if ss > 0:
            x    = torch.roll(x, (-ss, -ss), dims=(1, 2))
            mask = self._mask(Hp, Wp, ws, ss, x.device)
        else:
            mask = None
        win = _window_partition(x, ws).view(-1, ws * ws, C)
        win = self.attn(win, mask=mask).view(-1, ws, ws, C)
        x   = _window_reverse(win, ws, Hp, Wp)
        if ss > 0:
            x = torch.roll(x, (ss, ss), dims=(1, 2))
        if pad_r or pad_b:
            x = x[:, :H, :W, :].contiguous()
        x = shortcut + x.view(B, H * W, C)
        return x + self.mlp(self.norm2(x))


class TranBlock(nn.Module):
    """Two stacked Swin blocks: one regular, one shifted-window."""
    def __init__(self, channels: int, nhead: int = 8, window_size: int = 5):
        super().__init__()
        assert channels % nhead == 0, \
            f"channels ({channels}) must be divisible by nhead ({nhead})"
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(channels, nhead, window_size, shift_size=0),
            SwinTransformerBlock(channels, nhead, window_size,
                                 shift_size=window_size // 2),
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        seq = x.flatten(2).permute(0, 2, 1)
        for blk in self.blocks:
            seq = blk(seq, H, W)
        return seq.permute(0, 2, 1).view(B, C, H, W)


class PANet(nn.Module):
    """PANet FPN: 3 backbone levels → 4 output levels (D1–D4).
    Integrates FPCBAM on laterals and bottom-up path, TranBlocks throughout."""

    def __init__(self, in_channels_list: List[int], out_channels: int,
                 fpcbam_reduction: int = 16):
        super().__init__()
        C = out_channels
        self.laterals  = nn.ModuleList([
            BNSequential(ch, C, lateral=True) for ch in in_channels_list])

        self.td_fpcbam = nn.ModuleList([
            FPCBAM(gate_channels=C, reduction_ratio=fpcbam_reduction, use_pam=False),
            FPCBAM(gate_channels=C, reduction_ratio=fpcbam_reduction, use_pam=True),
            FPCBAM(gate_channels=C, reduction_ratio=fpcbam_reduction, use_pam=True),
        ])
        self.td_p5_tran = TranBlock(C, nhead=8)
        self.td_c3      = nn.ModuleList([C3Block(C, C, num_blocks=3) for _ in range(2)])
        self.td_fuse    = nn.ModuleList([BNSequential(C * 2, C) for _ in range(2)])

        self.bu_convs   = nn.ModuleList([BNSequential(C, C, stride=2) for _ in range(3)])
        self.bu_fuse    = nn.ModuleList([BNSequential(C * 2, C) for _ in range(3)])
        self.bu_tran    = nn.ModuleList([TranBlock(C, nhead=8) for _ in range(3)])
        self.bu_fpcbam  = nn.ModuleList([
            FPCBAM(gate_channels=C, reduction_ratio=fpcbam_reduction, use_pam=True)
            for _ in range(2)])

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        lats = [lat(f) for lat, f in zip(self.laterals, features)]
        lats = [self.td_fpcbam[i](lats[i]) for i in range(3)]

        # Top-down pathway
        p5_td    = self.td_p5_tran(lats[2])
        p5_up    = F.interpolate(p5_td, size=lats[1].shape[-2:], mode='nearest')
        p4_fused = self.td_fuse[1](torch.cat([lats[1], p5_up], dim=1))
        p4_td    = self.td_c3[1](p4_fused)
        p4_up    = F.interpolate(p4_td, size=lats[0].shape[-2:], mode='nearest')
        p3_fused = self.td_fuse[0](torch.cat([lats[0], p4_up], dim=1))
        d1       = self.td_c3[0](p3_fused)

        # Extra downsampled P5 levels for bottom-up
        p5_td_10 = F.avg_pool2d(p5_td, kernel_size=2, stride=2)
        p5_td_5  = F.avg_pool2d(p5_td_10, kernel_size=2, stride=2)

        # Bottom-up pathway
        d1_down = self.bu_convs[0](d1)
        bu1     = self.bu_fuse[0](torch.cat([p4_td, d1_down], dim=1))
        bu1     = self.bu_tran[0](bu1)
        d2      = self.bu_fpcbam[0](bu1)

        d2_down = self.bu_convs[1](d2)
        bu2     = self.bu_fuse[1](torch.cat([p5_td_10, d2_down], dim=1))
        bu2     = self.bu_tran[1](bu2)
        d3      = self.bu_fpcbam[1](bu2)

        d3_down = self.bu_convs[2](d3)
        bu3     = self.bu_fuse[2](torch.cat([p5_td_5, d3_down], dim=1))
        d4      = self.bu_tran[2](bu3)

        return [d1, d2, d3, d4]


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  Transformer Polygon Head
# ═══════════════════════════════════════════════════════════════════════════════

class PositionalEncoding2D(nn.Module):
    """Sinusoidal 2D positional encoding."""
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        b, c, h, w = tensor.shape
        pos_x = torch.arange(w, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(h, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        emb_x = (torch.cat([sin_inp_x.sin(), sin_inp_x.cos()], dim=-1)
                 .unsqueeze(0).repeat(h, 1, 1))
        emb_y = (torch.cat([sin_inp_y.sin(), sin_inp_y.cos()], dim=-1)
                 .unsqueeze(1).repeat(1, w, 1))
        emb   = (torch.cat([emb_y, emb_x], dim=-1)
                 .permute(2, 0, 1).unsqueeze(0).repeat(b, 1, 1, 1))
        return emb


class TransformerPolygonHead(nn.Module):
    """DETR-style detection head: encoder + decoder + anchor-relative polygon regression.

    Each head produces:
      - pred_logits: (B, Q, 1)  — binary text/non-text confidence (raw logits)
      - pred_coords: (B, Q, P, 2) — polygon vertices in [0,1]
      - ref_centers:  (B, Q, 2)    — reference anchor centres (for matching)
    """

    def __init__(self, in_channels: int, num_queries: int = 25,
                 num_points: int = 6, num_encoder_layers: int = 2,
                 num_decoder_layers: int = 6, nhead: int = 8,
                 dim_feedforward: int = 1024, dropout: float = 0.1,
                 conf_bias_init: float = -4.0):
        super().__init__()
        self.num_queries = num_queries
        self.num_points  = num_points
        self.in_channels = in_channels

        self.pos_enc = PositionalEncoding2D(in_channels // 2)
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.GroupNorm(32 if in_channels % 32 == 0 else 1, in_channels),
            nn.SiLU(inplace=True),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_channels, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=0.0, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.query_embed = nn.Embedding(self.num_queries, in_channels)

        # Learnable reference anchor points (initialised as random cluster centres)
        _centers   = torch.rand(self.num_queries, 1, 2)
        _spread    = (torch.rand(self.num_queries, num_points, 2) - 0.5) * 0.04
        _ref       = (_centers + _spread).clamp(0.02, 0.98)
        _ref_logit = torch.log(_ref / (1.0 - _ref))
        self.ref_points = nn.Parameter(
            _ref_logit.reshape(self.num_queries, num_points * 2))

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=in_channels, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.poly_head = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.GELU(),
            nn.Linear(in_channels, num_points * 2),
        )
        self.conf_head = nn.Linear(in_channels, 1)

        # ── Initialisation ───────────────────────────────────────────────
        nn.init.zeros_(self.poly_head[-1].weight)
        nn.init.zeros_(self.poly_head[-1].bias)
        nn.init.constant_(self.conf_head.bias, conf_bias_init)
        nn.init.trunc_normal_(self.query_embed.weight, std=0.02)
        self.query_embed.weight.data.clamp_(-0.06, 0.06)
        for layer in self.decoder.layers:
            if hasattr(layer, 'multihead_attn'):
                nn.init.xavier_uniform_(layer.multihead_attn.in_proj_weight)
                nn.init.zeros_(layer.multihead_attn.in_proj_bias)
                nn.init.xavier_uniform_(layer.multihead_attn.out_proj.weight)
                nn.init.zeros_(layer.multihead_attn.out_proj.bias)

    def forward(self, feat: torch.Tensor) -> dict:
        B = feat.shape[0]
        feat_proj = self.proj(feat)
        pos       = self.pos_enc(feat)
        feat_flat = feat_proj.flatten(2).transpose(1, 2)
        pos_flat  = pos.flatten(2).transpose(1, 2)

        memory = feat_flat + pos_flat
        memory = self.encoder(memory)

        tgt   = torch.zeros(B, self.num_queries, self.in_channels, device=feat.device)
        q_pos = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        for layer in self.decoder.layers:
            tgt = layer(tgt + q_pos, memory + pos_flat)
        out = tgt

        pred_logits = self.conf_head(out)
        # Anchor-relative: point = sigmoid(ref) + small delta
        ref   = self.ref_points.unsqueeze(0).sigmoid()
        delta = self.poly_head(out) * 0.1
        pred_coords = (ref + delta).clamp(0.0, 1.0)
        pred_coords = pred_coords.view(B, self.num_queries, self.num_points, 2)

        ref_xy = (self.ref_points.sigmoid()
                  .view(self.num_queries, self.num_points, 2).mean(dim=1))

        return {
            "pred_logits": pred_logits,
            "pred_coords": pred_coords,
            "ref_centers": ref_xy.unsqueeze(0).expand(B, -1, -1),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  Full Model
# ═══════════════════════════════════════════════════════════════════════════════

class FullModel(nn.Module):
    """DIPCYT-X40s: backbone → PANet (4 levels) → 4 TransformerPolygonHeads."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.backbone = TPHYOLOv5PlusPlus()
        if cfg.pretrained_backbone:
            try:
                from .pretrained import load_yolov5l_backbone_weights
                load_yolov5l_backbone_weights(
                    self.backbone,
                    cache_path=cfg.pretrained_cache_path,
                )
            except Exception as e:
                print(f"  ⚠ Pretrained backbone weight transfer skipped: {e}")

        self.fpn = PANet(
            in_channels_list=[256, 512, 512],
            out_channels=cfg.fpn_out_channels,
            fpcbam_reduction=cfg.fpcbam_reduction_ratio,
        )
        self.heads = nn.ModuleList([
            TransformerPolygonHead(
                in_channels=cfg.fpn_out_channels,
                num_queries=cfg.num_queries,
                num_points=cfg.num_points,
                num_encoder_layers=cfg.head_encoder_layers[i],
                num_decoder_layers=cfg.head_decoder_layers,
                nhead=cfg.head_nhead,
                dim_feedforward=cfg.head_dim_feedforward,
                dropout=cfg.head_dropout,
                conf_bias_init=cfg.conf_bias_init,
            )
            for i in range(4)
        ])

    def forward(self, x: torch.Tensor) -> dict:
        p3, p4, p5 = self.backbone(x)
        fpn_out = self.fpn([p3, p4, p5])

        logits_list, coords_list, ref_list = [], [], []
        for i, head in enumerate(self.heads):
            out = head(fpn_out[i])
            logits_list.append(out["pred_logits"])
            coords_list.append(out["pred_coords"])
            ref_list.append(out["ref_centers"])

        return {
            "pred_logits": torch.cat(logits_list, dim=1),
            "pred_coords": torch.cat(coords_list, dim=1),
            "ref_centers": torch.cat(ref_list, dim=1),
        }

    def enable_gradient_checkpointing(self, enable: bool = True) -> None:
        """Enables/disables gradient checkpointing for VRAM-constrained training."""
        self.cfg.enable_gradient_checkpointing = enable
        print(f"  [FullModel] Gradient checkpointing: {'ENABLED' if enable else 'DISABLED'}")

    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = True
