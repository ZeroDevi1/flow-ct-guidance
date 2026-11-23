"""2D 测量条件化流匹配模型（ControlFlowUNet）：冻结预训练 19.8M UNet 流先验，
挂载零卷积 2D ControlNet 条件分支，融合像素对齐 FBP、傅里叶切片覆盖分解与全局角度/噪声几何嵌入。

设计要点：
1. 冻结主干（base_model）全部参数；仅克隆其编码器（temb/inp/down/mid）作为可训练 ControlNet 分支，
   通过 15 个零初始化 1×1 卷积注入解码器跳连，外加 1 个零初始化 1×1 卷积注入 bottleneck。
2. 零初始化保证训练第 0 步时输出严格等于预训练无条件流先验 v_θ(x_t, t)（零破坏初始化）；
   同时 cond_stem 与 geom_mlp 使用常规初始化，从第 1 步起即向零卷积提供满幅度条件梯度。
3. 基于中央切片定理构造精确物理 2D 傅里叶覆盖掩码 M_freq(kx, ky)，在图像域将 x_t 与 x_fbp
   分解为已测频域分量与缺失楔/未测频域分量，像素严格对齐且仅耗单次 rfft2/irfft2（~0.1 ms）。
4. 推理时通过 BoundControlFlow 适配为标准 (x, t) -> v 接口，无缝复用 sample_posterior.py 的
   全部引导规则（none / dc / pigdm / hybrid / Poisson 噪声感知）与多样本不确定性评估。
"""
from __future__ import annotations

import copy
import math
from pathlib import Path

import torch
import torch.nn as nn

from src.models.unet import ResBlock, UNet, timestep_embedding
from src.physics.radon import CTOperator, poisson_ray_variance


def fourier_wedge_mask(
    n: int,
    angles: torch.Tensor,
    device=None,
    dtype=torch.float32,
    flat_pix: float = 1.15,
    sigma_pix: float = 1.0,
) -> torch.Tensor:
    """按平行束中央切片定理构造 2D rfft2 频域覆盖掩码 [1, 1, n, n//2 + 1] ∈ [0, 1]。

    对 radon.py 的旋转约定（theta=[[cos,-sin],[sin,cos]] 后沿行求和），投影角 θ 对应
    2D 频域中极角 φ = atan2(ky, kx) mod π == θ 的径向切片。
    用频域像素弧长距离 r_pix · d_π(φ, θ) 判定覆盖：
    - 全 180 角时 Nyquist 边缘半间距为 128·π/360 ≈ 1.117 像素，flat_pix=1.15 使全角掩码全 1；
    - 限角（la120/la90）在已测扇区全 1、缺失楔内随半径平滑降至 0；
    - 稀疏视角（sv30/sv60）低频 Nyquist 核全 1、高频呈精确径向辐条。
    """
    dev = device if device is not None else angles.device
    ky = torch.fft.fftfreq(n, device=dev, dtype=torch.float32) * n      # [-n/2, n/2)
    kx = torch.fft.rfftfreq(n, device=dev, dtype=torch.float32) * n     # [0, n/2]
    ky_g, kx_g = torch.meshgrid(ky, kx, indexing="ij")
    r_pix = torch.hypot(kx_g, ky_g)                                     # [n, n//2+1]
    phi = torch.remainder(torch.atan2(ky_g, kx_g), math.pi)             # [0, π)

    ang = torch.remainder(angles.to(device=dev, dtype=torch.float32), math.pi)  # [A]
    diff = (phi[..., None] - ang[None, None, :]).abs()
    d_pi = torch.minimum(diff, math.pi - diff).amin(dim=-1)             # [n, n//2+1]
    arc = r_pix * d_pi
    excess = (arc - flat_pix).clamp(min=0.0)
    mask = torch.exp(-0.5 * (excess / sigma_pix) ** 2)
    mask[0, 0] = 1.0
    return mask.to(dtype=dtype)[None, None]


def angle_coverage_vector(
    angles: torch.Tensor,
    n_bins: int = 180,
    noise_level: float = 0.0,
    device=None,
    dtype=torch.float32,
) -> torch.Tensor:
    """构造全局扫描几何+噪声描述向量 [n_bins + 2]：
    - 前 n_bins 维：在 [0, π) 等距 180 格上的角度覆盖指示（∈ [0, 1]）；
    - 第 n_bins 维：测量角度覆盖比例 A / n_bins；
    - 第 n_bins + 1 维：归一化噪声代理 log1p(mean_sigma_y2) / 5.0（无噪为 0）。
    """
    dev = device if device is not None else angles.device
    ang = torch.remainder(angles.to(device=dev, dtype=torch.float32), math.pi)
    idx = torch.round(ang / (math.pi / n_bins)).long() % n_bins
    hist = torch.zeros(n_bins, device=dev, dtype=torch.float32)
    hist.scatter_add_(0, idx, torch.ones_like(idx, dtype=torch.float32))
    hist = hist.clamp(max=1.0)
    frac = torch.tensor([float(angles.numel()) / float(n_bins)], device=dev, dtype=torch.float32)
    nlev = torch.tensor([float(noise_level)], device=dev, dtype=torch.float32)
    return torch.cat([hist, frac, nlev], dim=0).to(dtype=dtype)


@torch.no_grad()
def build_cond_context(
    op: CTOperator,
    y: torch.Tensor,
    physics_cfg: dict | None = None,
    x_fbp: torch.Tensor | None = None,
    freq_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """从测量 y 与算子 op 构造一次可复用的条件上下文（整个 ODE 采样循环只需算一次）。"""
    b = y.shape[0]
    if x_fbp is None:
        x_fbp = op.fbp(y)
    if freq_mask is None:
        freq_mask = fourier_wedge_mask(op.n, op.angles, device=y.device, dtype=torch.float32)
    else:
        freq_mask = freq_mask.to(device=y.device, dtype=torch.float32)

    x_fbp_f32 = x_fbp.float()
    x_fbp_meas = torch.fft.irfft2(torch.fft.rfft2(x_fbp_f32) * freq_mask, s=x_fbp.shape[-2:])

    noise_level = 0.0
    if physics_cfg is not None:
        i0 = physics_cfg.get("i0")
        sigma = float(physics_cfg.get("sigma", 0.0) or 0.0)
        mu_scale = float(physics_cfg.get("mu_scale", 0.06))
        if (i0 is not None and i0 > 0) or sigma > 0:
            var = poisson_ray_variance(y, i0=i0, sigma=sigma, mu_scale=mu_scale)
            mean_var = float(var.mean().item())
            noise_level = math.log1p(max(mean_var, 0.0)) / 5.0

    gvec = angle_coverage_vector(op.angles, n_bins=180, noise_level=noise_level, device=y.device, dtype=torch.float32)
    geom_vec = gvec[None].expand(b, -1).contiguous()
    return {
        "x_fbp": x_fbp_f32,
        "x_fbp_meas": x_fbp_meas,
        "freq_mask": freq_mask,
        "geom_vec": geom_vec,
    }


def _zero_conv(ch: int) -> nn.Conv2d:
    c = nn.Conv2d(ch, ch, 1)
    nn.init.zeros_(c.weight)
    nn.init.zeros_(c.bias)
    return c


def _encoder_skip_channels(base: UNet) -> tuple[list[int], int]:
    """推断 UNet 编码器 hs 列表中每一项的通道数与 mid 通道数。"""
    chs = [base.base_ch]
    cur = base.base_ch
    for blocks in base.down:
        if isinstance(blocks[0], nn.Conv2d):
            cur = blocks[0].out_channels
        else:
            for b in blocks:
                if isinstance(b, ResBlock):
                    cur = b.conv2.out_channels
        chs.append(cur)
    return chs, cur


class ControlFlowUNet(nn.Module):
    """冻结预训练 UNet + 零卷积 2D 测量条件化 ControlNet 分支。"""

    def __init__(self, base_model: UNet, cond_ch: int = 4, geom_dim: int = 182):
        super().__init__()
        self.base = base_model
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.base.eval()

        self.cond_ch = cond_ch
        self.geom_dim = geom_dim
        base_ch = self.base.base_ch
        temb_dim = self.base.temb_dim

        # 克隆预训练编码器 + bottleneck 作为可训练条件分支起点
        self.ctrl_temb = copy.deepcopy(self.base.temb)
        self.ctrl_inp = copy.deepcopy(self.base.inp)
        self.ctrl_down = copy.deepcopy(self.base.down)
        self.ctrl_mid = copy.deepcopy(self.base.mid)
        for m in (self.ctrl_temb, self.ctrl_inp, self.ctrl_down, self.ctrl_mid):
            for p in m.parameters():
                p.requires_grad_(True)

        # 像素对齐 2D 条件编码器：[x_fbp, x_fbp_meas, x_t_meas, x_t_miss] -> base_ch
        self.cond_stem = nn.Sequential(
            nn.Conv2d(cond_ch, 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, base_ch, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
        )
        # 全局几何+噪声嵌入：注入 ControlNet 时间条件向量 temb_ctrl
        self.geom_mlp = nn.Sequential(
            nn.Linear(geom_dim, temb_dim),
            nn.SiLU(),
            nn.Linear(temb_dim, temb_dim),
        )

        # 零初始化 1×1 注入卷积：第 0 步时残差严格为 0
        skip_chs, mid_ch = _encoder_skip_channels(self.base)
        self.zero_downs = nn.ModuleList([_zero_conv(c) for c in skip_chs])
        self.zero_mid = _zero_conv(mid_ch)

    def train(self, mode: bool = True):
        super().train(mode)
        # 始终保持冻结主干处于 eval 模式
        self.base.eval()
        return self

    def trainable_modules(self) -> dict[str, nn.Module]:
        return {
            "ctrl_temb": self.ctrl_temb,
            "ctrl_inp": self.ctrl_inp,
            "ctrl_down": self.ctrl_down,
            "ctrl_mid": self.ctrl_mid,
            "cond_stem": self.cond_stem,
            "geom_mlp": self.geom_mlp,
            "zero_downs": self.zero_downs,
            "zero_mid": self.zero_mid,
        }

    def trainable_parameters(self):
        for m in self.trainable_modules().values():
            yield from m.parameters()

    def trainable_state_dict(self) -> dict[str, dict]:
        return {k: m.state_dict() for k, m in self.trainable_modules().items()}

    def load_trainable_state_dict(self, state: dict[str, dict]):
        for k, m in self.trainable_modules().items():
            m.load_state_dict(state[k])

    def _run_encoder_mid(self, x, temb, inp, down, mid, add_inp: torch.Tensor | None = None):
        h = inp(x)
        if add_inp is not None:
            h = h + add_inp
        hs = [h]
        for blocks in down:
            if isinstance(blocks[0], nn.Conv2d):
                h = blocks[0](h)
            else:
                for b in blocks:
                    h = b(h, temb) if isinstance(b, ResBlock) else b(h)
            hs.append(h)
        for b in mid:
            h = b(h, temb) if isinstance(b, ResBlock) else b(h)
        return hs, h

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond_ctx: dict[str, torch.Tensor] | None = None,
        cond_scale: float = 1.0,
    ) -> torch.Tensor:
        if cond_ctx is None or cond_scale == 0.0:
            return self.base(x, t)

        t_emb_raw = timestep_embedding(t, self.base.base_ch)

        # 1) 冻结主干编码器 + bottleneck：训练期 x.requires_grad=False，可免存激活省显存；
        #    ΠGDM/DPS 推理期 x.requires_grad=True，自动保留对 x 的精确梯度。
        with torch.set_grad_enabled(x.requires_grad):
            temb_base = self.base.temb(t_emb_raw)
            hs_base, h_base = self._run_encoder_mid(
                x, temb_base, self.base.inp, self.base.down, self.base.mid
            )

        # 2) 傅里叶切片分解（rfft2 需在 float32 下运行以兼容 bf16 autocast）
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            x_f32 = x.float()
            fmask = cond_ctx["freq_mask"].float()
            x_t_meas = torch.fft.irfft2(torch.fft.rfft2(x_f32) * fmask, s=x.shape[-2:])
            x_t_miss = x_f32 - x_t_meas
            cond_in = torch.cat(
                [cond_ctx["x_fbp"].float(), cond_ctx["x_fbp_meas"].float(), x_t_meas, x_t_miss],
                dim=1,
            )

        # 3) 可训练 ControlNet 条件分支
        temb_ctrl = self.ctrl_temb(t_emb_raw) + self.geom_mlp(cond_ctx["geom_vec"].to(dtype=t_emb_raw.dtype))
        cond_feat = self.cond_stem(cond_in.to(dtype=x.dtype))
        hs_ctrl, h_ctrl = self._run_encoder_mid(
            x, temb_ctrl, self.ctrl_inp, self.ctrl_down, self.ctrl_mid, add_inp=cond_feat
        )

        # 4) 零卷积多尺度注入冻结主干解码器
        h = h_base + cond_scale * self.zero_mid(h_ctrl)
        hs = [hb + cond_scale * zc(hc) for hb, zc, hc in zip(hs_base, self.zero_downs, hs_ctrl)]
        for blocks in self.base.up:
            if isinstance(blocks[0], nn.Upsample):
                h = blocks[1](blocks[0](h))
            else:
                h = torch.cat([h, hs.pop()], dim=1)
                for b in blocks:
                    h = b(h, temb_base) if isinstance(b, ResBlock) else b(h)
        return self.base.out(h)


class BoundControlFlow(nn.Module):
    """将 ControlFlowUNet 与单批条件上下文绑定为标准 (x, t) -> v 接口，兼容 sample_posterior.py。"""

    def __init__(
        self,
        model: ControlFlowUNet,
        cond_ctx: dict[str, torch.Tensor],
        cond_scale: float = 1.0,
        cfg_scale: float = 1.0,
    ):
        super().__init__()
        self.model = model
        self.cond_ctx = cond_ctx
        self.cond_scale = float(cond_scale)
        self.cfg_scale = float(cfg_scale)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        v_cond = self.model(x, t, self.cond_ctx, cond_scale=self.cond_scale)
        if self.cfg_scale == 1.0:
            return v_cond
        v_uncond = self.model.base(x, t)
        return v_uncond + self.cfg_scale * (v_cond - v_uncond)


def load_control_flow(
    prior_ckpt: str | Path,
    control_ckpt: str | Path,
    device="cpu",
    use_ema: bool = True,
) -> tuple[ControlFlowUNet, dict]:
    """加载冻结先验 + ControlNet 条件分支权重（优先取 EMA）。"""
    from src.utils import load_ema

    base, mcfg = load_ema(prior_ckpt, device=device)
    ck = torch.load(control_ckpt, map_location=device)
    ctrl_cfg = ck.get("ctrl_cfg", {})
    model = ControlFlowUNet(
        base,
        cond_ch=int(ctrl_cfg.get("cond_ch", 4)),
        geom_dim=int(ctrl_cfg.get("geom_dim", 182)),
    ).to(device)
    key = "ema_trainable" if (use_ema and "ema_trainable" in ck) else "trainable"
    model.load_trainable_state_dict(ck[key])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, mcfg
