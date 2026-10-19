"""带时间嵌入的 U-Net（ADM/DDPM 风格精简版），用于流匹配速度场 v_θ(x_t, t) 与监督后处理基线。

规模通过 ``base_ch`` / ``ch_mult`` 调节：
* 冒烟:   base_ch=32,  ch_mult=(1,2,2)       ≈ 2M 参数
* 256² 正式: base_ch=96, ch_mult=(1,1,2,2,4), attn_res=(16,) ≈ 45M 参数（A100 单卡 ~13 h / 120k 步）
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """t ∈ [0,1] 连续时间 → 正弦嵌入 [B, dim]。"""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t.float()[:, None] * 1000.0 * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


def _groups(ch: int, max_groups: int = 32) -> int:
    """GroupNorm 分组数：不超过 max_groups 的最大整除数。"""
    for g in range(min(max_groups, ch), 0, -1):
        if ch % g == 0:
            return g
    return 1


class ResBlock(nn.Module):
    def __init__(self, cin, cout, temb_dim, dropout=0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(cin), cin)
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.temb = nn.Linear(temb_dim, cout * 2)
        self.norm2 = nn.GroupNorm(_groups(cout), cout)
        self.drop = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
        nn.init.zeros_(self.conv2.weight); nn.init.zeros_(self.conv2.bias)

    def forward(self, x, temb):
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.temb(F.silu(temb))[:, :, None, None].chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale) + shift
        h = self.conv2(self.drop(F.silu(h)))
        return self.skip(x) + h


class AttnBlock(nn.Module):
    def __init__(self, ch, heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(_groups(ch), ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)
        self.heads = heads
        nn.init.zeros_(self.proj.weight); nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, 3, self.heads, c // self.heads, h * w).unbind(1)
        out = F.scaled_dot_product_attention(q.transpose(-1, -2), k.transpose(-1, -2), v.transpose(-1, -2))
        out = out.transpose(-1, -2).reshape(b, c, h, w)
        return x + self.proj(out)


class UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base_ch=64, ch_mult=(1, 2, 2, 4), num_res=2,
                 attn_res=(16,), dropout=0.0, image_size=256, cond_ch=0):
        """cond_ch>0 时把条件图（如 FBP）在通道维拼接——FBPConvNet 基线复用此网络（t 固定为 0）。"""
        super().__init__()
        self.temb_dim = base_ch * 4
        self.temb = nn.Sequential(nn.Linear(base_ch, self.temb_dim), nn.SiLU(), nn.Linear(self.temb_dim, self.temb_dim))
        self.base_ch = base_ch
        self.inp = nn.Conv2d(in_ch + cond_ch, base_ch, 3, padding=1)

        self.down = nn.ModuleList()
        chs, ch, res = [base_ch], base_ch, image_size
        for i, m in enumerate(ch_mult):
            for _ in range(num_res):
                blocks = [ResBlock(ch, base_ch * m, self.temb_dim, dropout)]
                ch = base_ch * m
                if res in attn_res:
                    blocks.append(AttnBlock(ch))
                self.down.append(nn.ModuleList(blocks)); chs.append(ch)
            if i != len(ch_mult) - 1:
                self.down.append(nn.ModuleList([nn.Conv2d(ch, ch, 3, stride=2, padding=1)]))
                chs.append(ch); res //= 2

        self.mid = nn.ModuleList([ResBlock(ch, ch, self.temb_dim, dropout), AttnBlock(ch), ResBlock(ch, ch, self.temb_dim, dropout)])

        self.up = nn.ModuleList()
        for i, m in reversed(list(enumerate(ch_mult))):
            for _ in range(num_res + 1):
                blocks = [ResBlock(ch + chs.pop(), base_ch * m, self.temb_dim, dropout)]
                ch = base_ch * m
                if res in attn_res:
                    blocks.append(AttnBlock(ch))
                self.up.append(nn.ModuleList(blocks))
            if i != 0:
                self.up.append(nn.ModuleList([nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(ch, ch, 3, padding=1)]))
                res *= 2
        self.out = nn.Sequential(nn.GroupNorm(_groups(ch), ch), nn.SiLU(), nn.Conv2d(ch, out_ch, 3, padding=1))
        nn.init.zeros_(self.out[-1].weight); nn.init.zeros_(self.out[-1].bias)

    def forward(self, x, t, cond=None):
        temb = self.temb(timestep_embedding(t, self.base_ch))
        if cond is not None:
            x = torch.cat([x, cond], dim=1)
        h = self.inp(x)
        hs = [h]
        for blocks in self.down:
            if isinstance(blocks[0], nn.Conv2d):  # downsample
                h = blocks[0](h)
            else:
                for b in blocks:
                    h = b(h, temb) if isinstance(b, ResBlock) else b(h)
            hs.append(h)
        for b in self.mid:
            h = b(h, temb) if isinstance(b, ResBlock) else b(h)
        for blocks in self.up:
            if isinstance(blocks[0], nn.Upsample):
                h = blocks[1](blocks[0](h))
            else:
                h = torch.cat([h, hs.pop()], dim=1)
                for b in blocks:
                    h = b(h, temb) if isinstance(b, ResBlock) else b(h)
        return self.out(h)


def count_params(m: nn.Module) -> float:
    return sum(p.numel() for p in m.parameters()) / 1e6
