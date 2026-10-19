"""可微平行束 Radon 变换 / FBP / 共轭梯度数据一致性（纯 PyTorch，无外部 CT 库依赖）。

约定
----
* 图像 ``x``: [B, 1, N, N]，取值 [-1, 1]（-1 = 空气 / -1000 HU，+1 = +1000 HU）。
* 物理域衰减 ``u = mask * (x + 1) / 2`` ∈ [0, 1]，内切圆外置零（标准 CT 重建域）。
* 正弦图 ``y``: [B, 1, n_angles, N]，``y = R(u)``，R 是把图像旋转 θ 后沿行求和。
* ``CTOperator.forward(x)`` 是 x 的仿射映射：``A(x) = A_lin(x) + offset``，其中
  ``A_lin = R(mask * x / 2)``, ``offset = R(mask / 2)``。后验采样里的梯度、
  CG 数据一致性都通过 ``A_lin`` / ``A_lin_T`` 实现，后者是用 autograd 得到的精确伴随。

实现选择的说明
--------------
用 ``grid_sample`` 旋转 + 求和实现 Radon 是最简单的可微方案，256² × 180 角度一张图
在 A100 上约 3 ms。精度足够做 2D 方法学对比；Phase 2 若上锥束/真实几何，
再换 LEAP / torch-radon / ASTRA。
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _rot_grid(n: int, angles: torch.Tensor, device, dtype) -> torch.Tensor:
    """返回 [n_angles, N, N, 2] 的采样网格，使输出图像 = 输入图像旋转 angle。"""
    cos, sin = torch.cos(angles), torch.sin(angles)
    zeros = torch.zeros_like(cos)
    theta = torch.stack([torch.stack([cos, -sin, zeros], -1),
                         torch.stack([sin, cos, zeros], -1)], -2)  # [A, 2, 3]
    return F.affine_grid(theta.to(device=device, dtype=dtype), (angles.numel(), 1, n, n), align_corners=False)


def circle_mask(n: int, device=None, dtype=torch.float32, radius: float = 1.0) -> torch.Tensor:
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, n, device=device, dtype=dtype),
                            torch.linspace(-1, 1, n, device=device, dtype=dtype), indexing="ij")
    return ((xx ** 2 + yy ** 2) <= radius ** 2).to(dtype)[None, None]


def radon(u: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """u: [B,1,N,N] -> sinogram [B,1,A,N]。angles 单位弧度。"""
    b, _, n, _ = u.shape
    a = angles.numel()
    grid = _rot_grid(n, angles, u.device, u.dtype)  # [A,N,N,2]
    grid = grid[None].expand(b, a, n, n, 2).reshape(b * a, n, n, 2)
    rep = u.repeat_interleave(a, dim=0)  # [B*A,1,N,N]
    rot = F.grid_sample(rep, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    return rot.sum(dim=-2).reshape(b, 1, a, n)


def backproject(sino: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """近似伴随（旋转回去并涂抹），用于 FBP。精确伴随请用 CTOperator.A_lin_T。"""
    b, _, a, n = sino.shape
    smear = sino.reshape(b * a, 1, 1, n).expand(b * a, 1, n, n)
    grid = _rot_grid(n, -angles, sino.device, sino.dtype)
    grid = grid[None].expand(b, a, n, n, 2).reshape(b * a, n, n, 2)
    back = F.grid_sample(smear, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    return back.reshape(b, a, n, n).sum(dim=1, keepdim=True)


def ramp_filter(sino: torch.Tensor, kind: str = "ram-lak") -> torch.Tensor:
    """沿探测器轴做频域斜坡滤波，零填充到 2 倍长度避免环绕。"""
    n = sino.shape[-1]
    m = 2 * n
    freqs = torch.fft.rfftfreq(m, d=1.0, device=sino.device)  # cycles / pixel
    filt = freqs.abs().clone()
    if kind == "shepp-logan":
        w = torch.where(freqs == 0, torch.ones_like(freqs), torch.sin(math.pi * freqs) / (math.pi * freqs + 1e-12))
        filt = filt * w
    elif kind == "cosine":
        filt = filt * torch.cos(math.pi * freqs)
    spec = torch.fft.rfft(sino, n=m, dim=-1)
    out = torch.fft.irfft(spec * filt, n=m, dim=-1)[..., :n]
    return out


def fbp(sino: torch.Tensor, angles: torch.Tensor, kind: str = "ram-lak") -> torch.Tensor:
    """滤波反投影，返回物理域 u ∈ R[B,1,N,N]。角度假设在 [0, π) 内近似等距。"""
    filt = ramp_filter(sino, kind)
    recon = backproject(filt, angles)
    return recon * (math.pi / angles.numel())


class CTOperator:
    """稀疏视角 / 低剂量 CT 的正向模型（作用在归一化图像 x 上）。"""

    def __init__(self, n: int, n_angles: int = 180, angle_range: float = math.pi,
                 device="cpu", dtype=torch.float32, angle_offset: float = 0.0):
        self.n = n
        self.angles = torch.linspace(0, angle_range, n_angles + 1, device=device, dtype=dtype)[:-1] + angle_offset
        self.mask = circle_mask(n, device, dtype)
        self.device, self.dtype = device, dtype
        with torch.no_grad():
            self.offset = radon(self.mask / 2, self.angles)  # R(mask/2)，[1,1,A,N]
            # 离散化 + 双线性插值会让 FBP 幅值有百分之几的系统偏差，用均匀圆盘标定一次
            inner = circle_mask(n, device, dtype, radius=0.8).bool()
            rec = fbp(self.offset, self.angles)
            self.fbp_scale = float((self.mask / 2)[inner].mean() / rec[inner].mean().clamp(min=1e-6))
            # A_lin 的能量尺度，用于让正则权重与角度数无关：随机高斯 x 下 E||A x||² = tr(AᵀA)，
            # 故 ata_scale = tr(AᵀA)/像素数（AᵀA 平均对角元），aat_scale = tr(AAᵀ)/测量数。16 个样本，相对误差约 1%。
            g = torch.Generator(device="cpu").manual_seed(0)
            xr = torch.randn(16, 1, n, n, generator=g).to(device=device, dtype=dtype)
            e = (self.A_lin(xr) ** 2).flatten(1).sum(1).mean()
            self.ata_scale = float(e / (n * n))
            self.aat_scale = float(e / self.offset.numel())

    # ---- 基本映射 ----
    def x_to_u(self, x: torch.Tensor) -> torch.Tensor:
        return self.mask * (x + 1) / 2

    def u_to_x(self, u: torch.Tensor) -> torch.Tensor:
        return (u * 2 - 1) * self.mask + (-1.0) * (1 - self.mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return radon(self.x_to_u(x), self.angles)

    __call__ = forward

    def A_lin(self, x: torch.Tensor) -> torch.Tensor:
        return radon(self.mask * x / 2, self.angles)

    def A_lin_T(self, y: torch.Tensor) -> torch.Tensor:
        """精确伴随：∇_x <A_lin x, y>。"""
        x = torch.zeros(y.shape[0], 1, self.n, self.n, device=y.device, dtype=y.dtype, requires_grad=True)
        with torch.enable_grad():
            out = (self.A_lin(x) * y).sum()
            g, = torch.autograd.grad(out, x)
        return g

    # ---- 解析重建 ----
    def fbp(self, y: torch.Tensor, kind: str = "ram-lak") -> torch.Tensor:
        u = fbp(y, self.angles, kind) * self.fbp_scale
        return self.u_to_x(u).clamp(-1, 1)

    # ---- 噪声模型 ----
    def add_noise(self, y_clean: torch.Tensor, i0: float | None = None, sigma: float = 0.0,
                  mu_scale: float = 0.06, generator=None) -> torch.Tensor:
        """低剂量：Poisson 光子噪声（I0 入射光子数）。sigma: 附加高斯噪声（投影域，单位同 y）。
        mu_scale 把归一化线积分换算成真实衰减（水 ≈ 0.03/pixel @1.5mm，u=0.5）。"""
        y = y_clean
        if i0 is not None:
            lam = i0 * torch.exp(-mu_scale * y_clean)
            counts = torch.poisson(lam, generator=generator)
            counts = counts.clamp(min=1.0)
            y = -torch.log(counts / i0) / mu_scale
        if sigma > 0:
            y = y + sigma * torch.randn(y.shape, device=y.device, dtype=y.dtype, generator=generator)
        return y

    # ---- 数据一致性：(A_lin^T A_lin + λI) x = A_lin^T (y - offset) + λ x0 ----
    def cg_data_consistency(self, x0: torch.Tensor, y: torch.Tensor, lam: float = 1.0,
                            n_iter: int = 10, relative: bool = False) -> torch.Tensor:
        """relative=True 时 lam 以 ata_scale 为单位（λ = lam·tr(AᵀA)/N²），不同角度数下含义一致。"""
        if relative:
            lam = lam * self.ata_scale
        rhs = self.A_lin_T(y - self.offset) + lam * x0

        def Aop(v):
            return self.A_lin_T(self.A_lin(v)) + lam * v

        x = x0.clone()
        r = rhs - Aop(x)
        p = r.clone()
        rs = (r * r).flatten(1).sum(1)
        for _ in range(n_iter):
            Ap = Aop(p)
            alpha = rs / ((p * Ap).flatten(1).sum(1) + 1e-12)
            x = x + alpha.view(-1, 1, 1, 1) * p
            r = r - alpha.view(-1, 1, 1, 1) * Ap
            rs_new = (r * r).flatten(1).sum(1)
            if rs_new.max() < 1e-10:
                break
            p = r + (rs_new / (rs + 1e-12)).view(-1, 1, 1, 1) * p
            rs = rs_new
        return x


def weighted_cg_dc(op: "CTOperator", x0: torch.Tensor, y: torch.Tensor, w: torch.Tensor, lam: float = 1.0,
                   n_iter: int = 10, relative: bool = True) -> torch.Tensor:
    """逐角度加权的近端数据一致性：argmin_x ‖W^{1/2}(A x − y)‖² + λ‖x − x0‖²。

    op 为全角度算子；y 为全角度"伪测量"（已测行是实测值，未测行是补全值）；w [B, A] 为逐行权重
    （已测行通常为 1，未测行为补全置信度）。relative=True 时 λ 以 op.ata_scale 为单位，与 cg_data_consistency 一致。
    正规方程 (Aᵀ W A + λI) x = Aᵀ W (y − b) + λ x0 用 CG 求解；W 只对角作用在正弦图行上，保持对称正定。
    """
    if relative:
        lam = lam * op.ata_scale
    wv = w[:, None, :, None]
    rhs = op.A_lin_T(wv * (y - op.offset)) + lam * x0

    def Aop(v):
        return op.A_lin_T(wv * op.A_lin(v)) + lam * v

    x = x0.clone()
    r = rhs - Aop(x)
    p = r.clone()
    rs = (r * r).flatten(1).sum(1)
    for _ in range(n_iter):
        Ap = Aop(p)
        alpha = rs / ((p * Ap).flatten(1).sum(1) + 1e-12)
        x = x + alpha.view(-1, 1, 1, 1) * p
        r = r - alpha.view(-1, 1, 1, 1) * Ap
        rs_new = (r * r).flatten(1).sum(1)
        if rs_new.max() < 1e-10:
            break
        p = r + (rs_new / (rs + 1e-12)).view(-1, 1, 1, 1) * p
        rs = rs_new
    return x


def measured_mask(n_full: int, n_angles: int, angle_range: float = math.pi) -> torch.Tensor:
    """已测角度在 n_full 个全角度网格（[0, π) 等距）上的位置，[n_full] bool。

    要求测量角度落在全网格上：稀疏视角 n_full % n_angles == 0；限角时每个测量角间隔 = π/n_full。
    """
    step_full = math.pi / n_full
    meas = torch.linspace(0, angle_range, n_angles + 1)[:-1]
    idx = torch.round(meas / step_full).long()
    if not torch.allclose(idx.double() * step_full, meas.double(), atol=1e-6):
        raise ValueError(f"测量角度不在 {n_full} 角全网格上：n_angles={n_angles}, range={angle_range}")
    m = torch.zeros(n_full, dtype=torch.bool)
    m[idx] = True
    return m


def interp_missing_rows(y_full: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """把未测行用相邻已测行沿 θ 的环形线性插值填上（不学习的补全基线）。

    y_full [B,1,A,D]（未测行内容被忽略），mask [A] bool。环形在 [0, π) 上闭合时使用 s(θ+π, r) = s(θ, −r)：
    越过 π 的那个邻居行要沿探测器翻转。限角时缺口两端分别是最后一个已测角与"翻转后的第一个已测角"。
    """
    a = mask.numel()
    idx = mask.nonzero()[:, 0].tolist()
    out = y_full.clone()
    for j in range(a):
        if mask[j]:
            continue
        lo = max((i for i in idx if i < j), default=idx[-1] - a)  # 可能来自上一圈（负数表示 θ−π）
        hi = min((i for i in idx if i > j), default=idx[0] + a)  # 可能来自下一圈（θ+π）
        w = (j - lo) / (hi - lo)
        r_lo = y_full[..., lo % a, :].flip(-1) if lo < 0 else y_full[..., lo, :]
        r_hi = y_full[..., hi % a, :].flip(-1) if hi >= a else y_full[..., hi, :]
        out[..., j, :] = (1 - w) * r_lo + w * r_hi
    return out


def sparse_angles(n_full: int, n_keep: int, device="cpu") -> torch.Tensor:
    """从 n_full 个等距角度里均匀保留 n_keep 个。"""
    idx = torch.linspace(0, n_full - 1, n_keep).round().long()
    full = torch.linspace(0, math.pi, n_full + 1, device=device)[:-1]
    return full[idx]
