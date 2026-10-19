"""角度序列补全器 C_φ：在 θ 轴上补全正弦图缺失行，并给出逐角度的不确定度 log σ²_θ。

输入 / 输出（设计见 docs/02_phase1b_angle_ssm_design.md §3）
    s_pred [B,1,A,D]  当前干净估计 x̂₁ 的全角度正弦图 A_full(x̂₁)
    y_full [B,1,A,D]  实测正弦图放到全角度网格上（未测行为 0）
    mask   [A] bool   哪些角度是实测的（一个 batch 共用一种采样模式）
    t      [B]        流时间，t=0 为噪声、t=1 为数据
    返回 (ŷ [B,1,A,D], log σ² [B,A])，ŷ 的已测行被强制等于 y_full。

角度混合器（消融用，mixer 参数）
    ssm  : 双向选择性状态空间模型（Mamba 式，Δ/B/C 随输入变化），环形边界通过"扫两圈取第二圈"实现
    conv : 沿 θ 的环形空洞卷积
    attn : 沿 θ 的自注意力（环形相对位置由正弦位置编码近似）
三种混合器层数、宽度一致，参数量接近，用于回答"为什么用 SSM"。

序列构造
    延拓到 2A 行：s(θ+π, r) = s(θ, −r)，得到 [0, 2π) 上严格周期的序列；输出取前 A 行。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.unet import timestep_embedding


def selective_scan(u: torch.Tensor, delta: torch.Tensor, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor,
                   chunk: int = 8) -> torch.Tensor:
    """对角选择性 SSM 的扫描：h_k = exp(Δ_k A) ⊙ h_{k-1} + Δ_k B_k u_k，y_k = ⟨C_k, h_k⟩。

    形状：u, delta [B, L, Dm]；A [Dm, N]（负实数）；B, C [B, L, N]；返回 [B, L, Dm]。
    实现：长度 chunk 的块内用累积乘积的闭式解并行计算，块间顺序传递状态。块内 log 衰减被钳到 ≥ −5，
    使 1/∏a ≤ e^{5·chunk}，chunk=8 时 ≤ e^40，在 float32 范围内（≤3e38）。
    数值上与逐步循环一致，见 tests/test_angle_ssm.py。
    """
    # 块内闭式解的动态范围达 e^40，bf16 只有 8 位尾数会严重失精度：扫描一律在 float32 中进行
    with torch.autocast(device_type=u.device.type, enabled=False):
        return _scan_fp32(u.float(), delta.float(), A.float(), B.float(), C.float(), chunk).to(u.dtype)


def _scan_fp32(u, delta, A, B, C, chunk):
    b, L, dm = u.shape
    n = A.shape[1]
    log_a = (delta[..., None] * A).clamp(min=-5.0)  # [B, L, Dm, N]
    bu = (delta * u)[..., None] * B[:, :, None, :]  # [B, L, Dm, N]
    h = u.new_zeros(b, dm, n)
    ys = []
    for s0 in range(0, L, chunk):
        la = log_a[:, s0:s0 + chunk]
        cum = la.cumsum(1)  # 块内 log ∏_{i≤k} a_i
        P = cum.exp()
        hk = P * (h[:, None] + (bu[:, s0:s0 + chunk] / P).cumsum(1))  # [B, l, Dm, N]
        ys.append((hk * C[:, s0:s0 + chunk, None, :]).sum(-1))
        h = hk[:, -1]
    return torch.cat(ys, 1)


class SSMBlock(nn.Module):
    """双向选择性 SSM 混合器：前向、后向各一次扫描后相加，带门控与残差。"""

    def __init__(self, d: int, d_state: int = 16, expand: int = 2):
        super().__init__()
        di = d * expand
        self.inp = nn.Linear(d, 2 * di)
        self.conv = nn.Conv1d(di, di, 4, padding=3, groups=di)  # Mamba 中扫描前的短因果卷积
        self.x_proj = nn.Linear(di, 2 * d_state + 1)  # → B, C, Δ（Δ 用低秩 1 再广播，减少参数）
        self.dt = nn.Linear(1, di)
        # S4D-real 初始化：A_n = −(n+1)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1).float()).repeat(di, 1))
        self.D = nn.Parameter(torch.ones(di))
        self.out = nn.Linear(di, d)
        self.d_state = d_state

    def _scan(self, x: torch.Tensor) -> torch.Tensor:
        L = x.shape[1]
        xc = F.silu(self.conv(x.transpose(1, 2))[..., :L].transpose(1, 2))
        bcd = self.x_proj(xc)
        Bm, Cm, dt = bcd.split([self.d_state, self.d_state, 1], dim=-1)
        delta = F.softplus(self.dt(dt))
        y = selective_scan(xc, delta, -self.A_log.exp(), Bm, Cm)
        return y + xc * self.D

    def forward(self, x):
        xz = self.inp(x)
        h, z = xz.chunk(2, dim=-1)
        y = self._scan(h) + self._scan(h.flip(1)).flip(1)
        return self.out(y * F.silu(z))


class ConvBlock(nn.Module):
    """环形空洞卷积混合器（消融对照）。核长 3、逐层空洞 1/3/9/27，4 层感受野 81 行；参数量与 SSMBlock 对齐。"""

    def __init__(self, d: int, dilation: int = 1):
        super().__init__()
        self.pad = dilation
        self.conv = nn.Conv1d(d, 2 * d, 3, dilation=dilation)
        self.out = nn.Linear(d, d)

    def forward(self, x):
        h = F.pad(x.transpose(1, 2), (self.pad, self.pad), mode="circular")
        a, g = self.conv(h).chunk(2, dim=1)
        return self.out((a * torch.sigmoid(g)).transpose(1, 2))


class AttnBlock1D(nn.Module):
    """沿 θ 的多头自注意力 + 窄 MLP 混合器（消融对照）；MLP 宽度取 d，使参数量与 SSMBlock 对齐。"""

    def __init__(self, d: int, heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))

    def forward(self, x):
        h = self.attn(x, x, x, need_weights=False)[0]
        return h + self.mlp(x + h)


class AngleCompleter(nn.Module):
    def __init__(self, n_det: int = 256, d: int = 256, n_layers: int = 4, mixer: str = "ssm",
                 t_cond: bool = True, det_ch: int = 16, det_down: int = 4, d_state: int = 16, expand: int = 2):
        """d_state / expand 只作用于 ssm 混合器；扫描中间张量为 [B, 4A, d·expand, d_state]，显存随二者线性增长。"""
        super().__init__()
        self.mixer_name, self.t_cond, self.n_det = mixer, t_cond, n_det
        # 探测器方向编码：3 通道（s_pred, y⊙m, m）→ det_ch 通道，r 方向下采样 det_down 倍，再展平到 d
        self.det_enc = nn.Sequential(
            nn.Conv1d(3, det_ch, 5, padding=2), nn.SiLU(),
            nn.Conv1d(det_ch, det_ch, det_down * 2 - 1, stride=det_down, padding=det_down - 1), nn.SiLU())
        self.tok = nn.Linear(det_ch * (n_det // det_down), d)
        self.pos = nn.Linear(4, d)  # 环形角度位置：sin/cos(θ)、sin/cos(2θ)，θ ∈ [0, 2π)
        self.temb = nn.Sequential(nn.Linear(64, d), nn.SiLU(), nn.Linear(d, 2 * d * n_layers))
        blocks = []
        for i in range(n_layers):
            if mixer == "ssm":
                blocks.append(SSMBlock(d, d_state=d_state, expand=expand))
            elif mixer == "conv":
                blocks.append(ConvBlock(d, dilation=3 ** i))
            elif mixer == "attn":
                blocks.append(AttnBlock1D(d))
            else:
                raise ValueError(mixer)
        self.blocks = nn.ModuleList(blocks)
        self.norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(n_layers)])
        self.dec = nn.Linear(d, n_det + 1)  # 每行：残差 δs（n_det 维）+ log σ²
        nn.init.zeros_(self.dec.weight); nn.init.zeros_(self.dec.bias)  # 初始 ŷ = s_pred、σ² = 1

    def forward(self, s_pred, y_full, mask, t):
        b, _, a, dn = s_pred.shape
        # 归一化：正弦图量级随物体尺寸变化，用已测行的均方根做逐样本尺度
        m = mask.to(s_pred.dtype)
        scale = (y_full[:, 0] ** 2 * m[None, :, None]).sum((1, 2)) / (m.sum() * dn)
        scale = scale.clamp(min=1e-6).sqrt().view(b, 1, 1, 1)
        feats = torch.cat([s_pred / scale, y_full / scale, m.view(1, 1, a, 1).expand(b, 1, a, dn)], 1)
        feats = torch.cat([feats, feats.flip(-1)], dim=2)  # 周期延拓到 2A 行 [B,3,2A,D]
        h = self.det_enc(feats.permute(0, 2, 1, 3).reshape(b * 2 * a, 3, dn))
        h = self.tok(h.reshape(b, 2 * a, -1))
        theta = torch.arange(2 * a, device=h.device, dtype=h.dtype) * (math.pi / a)
        h = h + self.pos(torch.stack([theta.sin(), theta.cos(), (2 * theta).sin(), (2 * theta).cos()], -1))
        if self.t_cond:
            film = self.temb(timestep_embedding(t, 64)).view(b, len(self.blocks), 2, -1)
        if self.mixer_name == "ssm":
            h = torch.cat([h, h], 1)  # 扫两圈：第二圈的每个位置都"看过"整圈，近似环形边界
        for i, (blk, nrm) in enumerate(zip(self.blocks, self.norms)):
            z = nrm(h)
            if self.t_cond:
                z = z * (1 + film[:, i, 0, None]) + film[:, i, 1, None]
            h = h + blk(z)
        if self.mixer_name == "ssm":
            h = h[:, 2 * a:]
        o = self.dec(h[:, :a])  # 只取物理上的前 A 行
        y_hat = s_pred + o[..., :dn][:, None] * scale
        logvar = o[..., dn]
        y_hat = torch.where(mask.view(1, 1, a, 1), y_full, y_hat)
        return y_hat, logvar


def count_params(m: nn.Module) -> float:
    return sum(p.numel() for p in m.parameters()) / 1e6
